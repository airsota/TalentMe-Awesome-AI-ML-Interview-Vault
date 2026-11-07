# 👁️ Multimodal & Diffusion · 多模态与扩散模型 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **37 cards · 185 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. ASR (Whisper) · 语音识别 ASR

> ASR maps an audio sequence $x = (x_1, \dots, x_T)$ (~100 frames/s) to text $y = (y_1, \dots, y_L)$; the core difficulty is length mismatch ($T \gg L$) and alignment learning.

**Interview questions**

1. How is CTC computed; the role of blank; how does it handle length mismatch?
2. CTC vs attention decoder: pros and cons of each?
3. Whisper's multi-task training format, and where its weak supervision comes from?
4. What does 680k hours buy Whisper, and what are its limits?
5. Why is CTC preferred for streaming ASR and attention for offline — what's the basis?

<details>
<summary>中文题目</summary>

1. CTC 损失怎么算?blank 的作用?如何解决长度不对齐?
2. CTC 与注意力解码器(attention decoder)各有什么优缺点?
3. Whisper 的多任务训练格式是什么?弱监督数据从哪来?
4. 68 万小时数据带来哪些能力?Whisper 的局限是什么?
5. 为什么说 CTC 适合流式、注意力适合离线?依据是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/asr?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=asr)

---

## 2. Audio Representation · 音频表示

> Audio representation has three levels, from raw waveform to model-ready features, determining input dimension and information retained: 1.

**Interview questions**

1. How is a mel-spectrogram computed; what do 80 bins, 25 ms, 10 ms mean?
2. Pros and cons of waveform vs mel-spectrogram vs discrete tokens?
3. How do neural codecs tokenize audio; typical bitrate and compression ratio?
4. Why do speech LLMs consume discrete audio tokens instead of regressing mels?
5. Why does RVQ reach ~1/100 bitrate with acceptable reconstruction quality?

<details>
<summary>中文题目</summary>

1. 梅尔谱如何计算?80 bins、25ms 窗、10ms 帧移这些数字的意义?
2. 波形 / 梅尔谱 / 离散 token 三种表示的优缺点对比?
3. 神经音频编解码器如何把波形变成离散 token?典型码率与压缩比?
4. 为什么语音 LLM 用离散音频 token 而不是直接回归梅尔谱?
5. RVQ 为什么能把码率压到约 1/100 而重建质量仍可接受?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/audio-representation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=audio-representation)

---

## 3. Classifier-Free Guidance · Classifier-Free Guidance (CFG)

> Classifier-Free Guidance linearly extrapolates between a conditional and an unconditional prediction so that sampling amplifies the condition $c$.

**Interview questions**

1. Write the CFG sampling formula and explain every symbol.
2. What is the typical CFG dropout probability, and how does one network learn both branches?
3. How does increasing w affect adherence, diversity and oversaturation, and why?
4. How does classifier-free guidance relate to classifier guidance; pros and cons of each?
5. What do w=1 and w=0 correspond to, and what is CFG's extra inference cost?

<details>
<summary>中文题目</summary>

1. 写出 CFG 采样公式 ε̂ = ε_θ(x_t,∅) + w·(ε_θ(x_t,c) − ε_θ(x_t,∅)), 并解释各符号含义?
2. CFG dropout 概率一般取多少? 训练时如何让条件与无条件分支共用同一网络?
3. w 增大对提示一致性、多样性、过饱和分别有什么影响? 为什么?
4. Classifier-Free Guidance 与分类器引导 (classifier guidance) 的关系? 各自优缺点?
5. w=1 与 w=0 分别对应什么? CFG 在推理期的额外计算开销是多少?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/cfg-guidance?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cfg-guidance)

---

## 4. CLIP Applications · CLIP 应用

> CLIP learns a shared image-text semantic space, making it a zero-shot multimodal foundation with four application families: ① zero-shot retrieval/classification — text→image search, 76.2% zero-shot top-1 on ImageNet;

**Interview questions**

1. CLIP's role in Stable Diffusion? Why do text-to-image models universally reuse the CLIP text tower?
2. How to use CLIP embeddings for RAG/retrieval? How is joint image-text retrieval done?
3. Limitations of CLIP zero-shot classification? What does the gap to linear probe reveal?
4. CLIP score for generation evaluation: mechanism? How well does it correlate with human judgment?
5. CLIP retrieval failure modes (fine-grained, abstract semantics, long-tail)? Mitigations?

<details>
<summary>中文题目</summary>

1. CLIP 在 Stable Diffusion 里起什么作用?为什么文生图普遍用 CLIP 文本塔?
2. CLIP embedding 在 RAG/检索场景怎么用?图文联合检索怎么做?
3. CLIP 零样本分类的局限?与线性探测 (linear probe) 的差距说明什么?
4. CLIP score 评估生成质量的原理?与人工评估的相关性如何?
5. CLIP 检索的失败模式 (细粒度/抽象语义/长尾) 有哪些?如何缓解?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/clip-applications?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=clip-applications)

---

## 5. CLIP Two-Tower Architecture · CLIP 双塔架构

> CLIP (Contrastive Language-Image Pretraining) is OpenAI's contrastive multimodal model with a two-tower design: an image encoder (ViT such as ViT-B/32, or ResNet) and a text encoder (Transformer), jointly trained on 400M image-text pairs (WIT) to embed images…

**Interview questions**

1. Which encoders do CLIP's two towers use, and how do both towers share one contrastive space?
2. CLIP vs VLM: CLIP only aligns embeddings, why can't it directly chat?
3. The full zero-shot classification pipeline of CLIP? Role of the prompt template (“a photo of a {label}”)?
4. Why does CLIP use a symmetric loss (image→text + text→image)? What would one-directional training lose?
5. CLIP's role in Stable Diffusion: which tower is used, frozen or trainable?

<details>
<summary>中文题目</summary>

1. CLIP 双塔分别用什么编码器?两个 tower 如何共享同一个对比空间?
2. CLIP 和 VLM 的区别?CLIP 只会做嵌入对齐, 为什么不能直接对话?
3. CLIP 零样本分类的完整流程?prompt 模板 (“a photo of a {label}”) 的作用?
4. 为什么 CLIP 用对称损失 (图→文 + 文→图双向)?只算单向会怎样?
5. CLIP 在 Stable Diffusion 里扮演什么角色?用的是哪一塔、冻结还是可训练?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/clip-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=clip-architecture)

---

## 6. ColPali Visual Retrieval · ColPali Late-Interaction 视觉检索

> ColPali feeds PDF pages directly into a VLM (PaliGemma) to obtain patch-level vectors, encodes the query into token-level vectors with an LLM, and scores with ColBERT-style Late-Interaction MaxSim: $$\text{sim}(q, d) = \sum_i \max_j \langle q_i, d_j \rangle$$…

**Interview questions**

1. Write ColPali/ColBERT's Late-Interaction MaxSim scoring formula and explain each term?
2. ColPali vs the 'OCR + text retrieval' pipeline: differences and gains (with metrics)?
3. Why does ColPali use VLM patch-level vectors instead of CLIP's global image vector?
4. Late-interaction vs dual-tower global-vector retrieval: precision/latency tradeoff?
5. How are ColPali document vectors precomputed offline? Online retrieval and rerank pipeline?

<details>
<summary>中文题目</summary>

1. 写出 ColPali/ColBERT 的 Late-Interaction MaxSim 打分公式并解释各项含义?
2. ColPali 与传统 "OCR + 文本检索" 管线的区别与优势 (含指标提升)?
3. 为什么 ColPali 用 VLM 的 Patch 级向量而非 CLIP 的全局图像向量?
4. Late-Interaction 与双塔全局向量检索的精度/延迟权衡?
5. ColPali 的文档向量如何离线预计算? 在线检索与重排流程如何设计?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/colpali?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=colpali)

---

## 7. DDPM Forward/Reverse Process · DDPM 前向/反向过程

> DDPM (Denoising Diffusion Probabilistic Models) builds generation on two Markov chains: a forward process $q$ that gradually adds noise to an image until it becomes standard Gaussian over $T$ steps, and a reverse process $p_\theta$ that learns to denoise step…

**Interview questions**

1. Write the closed-form forward process q(x_t|x_0) and define the cumulative schedule ᾱ_t.
2. Whiteboard: derive the training loss E‖ε−ε_θ(x_t,t)‖² from the ELBO; how each Gaussian KL term simplifies.
3. Write the DDPM reverse-sampling update; why is no noise added at the final step, and what is σ_t?
4. Relation between α_t, β_t and ᾱ_t; why can forward sampling jump to any t in one step?
5. How do DDPM, GAN and VAE compare on training stability, sample diversity and sampling speed?

<details>
<summary>中文题目</summary>

1. 写出 DDPM 前向过程的封闭形式 q(x_t|x_0) 与累积噪声调度 ᾱ_t 的定义?
2. 白板推导: 从变分下界 ELBO 到训练损失 L = E[‖ε − ε_θ(x_t,t)‖²], 每项高斯 KL 如何化简?
3. 写出 DDPM 采样迭代公式, 末步为什么不加噪声? σ_t 取什么?
4. α_t、β_t、ᾱ_t 的关系? 为什么前向可以一步采样到任意时刻 t?
5. DDPM 相比 GAN 与 VAE 在训练稳定性、样本多样性、采样速度上的差异?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/ddpm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ddpm)

---

## 8. Diffusion Training Objectives (ε/v-pred, Zero-SNR) · 扩散训练目标 (ε/v-pred 与 Zero-SNR)

> Diffusion training objectives share the same underlying posterior mean but differ in numerical behavior across signal-to-noise ratios (SNR).

**Interview questions**

1. Difference between ε-pred and v-pred; definitions of α_t and σ_t in v = α_t·ε − σ_t·x_0?
2. What is Zero-SNR, and why does it require v-prediction?
3. Write the interchange relations between v-pred, x_0 and ε; how to recover x_0 from v_θ?
4. Typical CFG dropout probability; can CFG work without it?
5. Why do SD2/SDXL/Flux use v-pred over ε-pred, and how does loss weighting vary across timesteps?

<details>
<summary>中文题目</summary>

1. ε-pred 与 v-pred 的区别? v = α_t·ε − σ_t·x_0 中 α_t、σ_t 的定义?
2. 什么是 Zero-SNR? 为什么 Zero-SNR 调度下必须用 v-pred?
3. 写出 v-pred 与 x_0、ε 的互换关系, 采样时如何从 v_θ 恢复 x_0?
4. CFG dropout 概率一般取多少? 没有它 CFG 还能用吗?
5. 为什么 SD2/SDXL/Flux 用 v-pred 而不是 ε-pred? 各时间步损失权重如何变化?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/diffusion-training-loss?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=diffusion-training-loss)

---

## 9. Diffusion vs GAN vs Autoregressive · 扩散 vs GAN vs 自回归

> The three major generation families trade off quality/diversity/speed. GANs play a zero-sum game between generator and discriminator — one forward pass is fastest, but adversarial training is unstable and prone to mode collapse.

**Interview questions**

1. Compare diffusion vs GAN on quality, diversity and speed; why is diffusion the T2I SOTA?
2. Write and explain the FID formula; why does it not fully capture diversity?
3. Causes and detection of GAN mode collapse; how does diffusion naturally avoid it?
4. Pros and cons of autoregressive vs diffusion generation; why is AR being displaced on images?
5. For real-time use (mobile SR/filters), GAN or distilled diffusion — how to trade off latency and quality?

<details>
<summary>中文题目</summary>

1. 扩散 vs GAN 在质量、多样性、速度上的对比? 为什么文生图 SOTA 用扩散?
2. 写出 FID 公式并解释其含义, 为什么它不能完全代表多样性?
3. GAN 模式坍塌的成因与检测方法? 扩散如何天然规避这一问题?
4. 自回归生成相比扩散的优缺点? 为什么自回归在图像上逐渐被取代?
5. 实时场景(如手机端超分/滤镜)选 GAN 还是蒸馏扩散? 各自延迟与质量如何权衡?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/diffusion-vs-gan?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=diffusion-vs-gan)

---

## 10. DiT & 3D Causal VAE · DiT 与 3D Causal VAE

> DiT (Diffusion Transformer) replaces the U-Net with a pure Transformer as the diffusion backbone — the underlying architecture of Sora, SD3 and FLUX.

**Interview questions**

1. How does DiT patchify work; given a $h\times w\times c$ latent and patch $p$, how many tokens?
2. How does AdaLN inject $t$/condition, and why does AdaLN-Zero zero-init the MLP output?
3. What are Sora's 3D causal VAE spatio-temporal compression ratios; why causal in time?
4. DiT vs U-Net diffusion backbones; why did SD3/FLUX also switch to DiT?
5. How does the 3D causal VAE support arbitrary-length video training and inference?

<details>
<summary>中文题目</summary>

1. DiT 的 Patchify 如何工作?给定 $h \times w \times c$ 潜图与 patch $p$,token 数是多少?
2. AdaLN 如何注入时间步 $t$ 与条件?AdaLN-Zero 为什么把 MLP 输出初始化为 0?
3. Sora 3D causal VAE 的时空压缩比是多少?为什么时间维用因果卷积?
4. DiT 相比 U-Net 扩散主干的优劣势?为什么 SD3/FLUX 也转向 DiT?
5. 3D causal VAE 如何支持任意长度视频的训练与推理?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/dit-3dvae?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dit-3dvae)

---

## 11. Embodied AI (cross-module) · 具身智能 (跨模块)

> embodied AI closes the loop between multimodal perception (VLM) and decision-making (RL): a world model provides imagined rollouts for MPC planning, policies are optimized by RL, and Diffusion Policy generates continuous action trajectories.

**Interview questions**

1. How do perception (VLM) and decision (RL) close the loop in embodied AI? Where does the world model sit?
2. How does MPC plan with imagined world-model rollouts? Difference from pure policy learning?
3. Why is Diffusion Policy well suited to robot action generation?
4. Impact of world-model error in the embodied loop? How to mitigate compounding error?
5. Where does embodied training data come from? Teleoperation/simulation vs real-data tradeoffs?

<details>
<summary>中文题目</summary>

1. 具身智能中感知 (VLM) 与决策 (RL) 如何闭环? 世界模型在链路中处于什么位置?
2. MPC 用世界模型做想象规划的原理? 与纯策略学习的区别?
3. Diffusion Policy 为什么适合机器人动作生成?
4. 世界模型误差在具身闭环中的影响? 如何缓解复合误差?
5. 具身训练数据从哪来? 遥操作/仿真与真实数据的各自优劣?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/embodied-rl?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=embodied-rl)

---

## 12. High-Res & Dynamic Tokens · AnyRes 切块 / 动态 Token

> High-resolution input is the key to VLM performance on OCR, documents and small objects; two mainstream schemes: ① LLaVA-1.5/1.6 AnyRes — split the high-res image into an $n_h \times n_w$ grid of 336×336 cells, run each cell through the vision encoder for 576…

**Interview questions**

1. How does AnyRes work? How are grids split and what does the global thumbnail do?
2. LLaVA-1.6 tiers (1/2/4 grids): how many visual tokens does each produce?
3. Qwen2-VL dynamic resolution vs AnyRes fixed grids: differences?
4. Why does high resolution drastically improve OCR/document ability? At what cost?
5. How does visual token count affect LLM attention cost (quadratic in sequence length)?

<details>
<summary>中文题目</summary>

1. AnyRes 的原理?高分辨率图怎么切块、全局缩略图干什么用?
2. LLaVA-1.6 各档位 (1/2/4 网格) 各产生多少 visual token?
3. Qwen2-VL 动态分辨率与 AnyRes 固定切块的区别?
4. 为什么高分辨率能显著提升 OCR/文档能力?代价是什么?
5. 视觉 token 数如何影响 LLM 的 attention 成本 (随长度平方增长)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/high-res-dynamic-token?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=high-res-dynamic-token)

---

## 13. InfoNCE Contrastive Loss · InfoNCE 对比损失

> InfoNCE (Info Noise-Contrastive Estimation) is the core loss of contrastive learning (CLIP/SimCLR/MoCo) — pull positives together and push all in-batch negatives apart:…

**Interview questions**

1. Write the InfoNCE loss and explain every symbol? How is it related to cross-entropy?
2. How to pick temperature τ? What happens when τ is too small or too large (numeric example)?
3. Why does a larger batch size help? Where do negatives come from?
4. InfoNCE vs triplet loss and vanilla NCE: key differences?
5. How does temperature interact with gradients in CLIP's symmetric loss? Handling hard negatives?

<details>
<summary>中文题目</summary>

1. 写出 InfoNCE 损失公式并解释每个符号?它与交叉熵是什么关系?
2. 温度 τ 怎么选?τ 过小/过大会怎样 (给出数值例)?
3. 为什么 batch size 越大效果越好?负样本从哪里来?
4. InfoNCE 与三元组损失、原始 NCE 的区别?
5. CLIP 对称损失中温度如何影响梯度?遇到 hard negatives 怎么办?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/contrastive-loss?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=contrastive-loss)

---

## 14. Interleaved Training Data · 图文交错训练数据

> Interleaved training data is the data pyramid for VLM pretraining, organized as image-text pairs → interleaved documents → video; the mixing ratio effectively sets the capability ceiling.

**Interview questions**

1. What is the three-tier data pyramid for VLM pretraining and the role of each tier?
2. How does the mixing ratio of pairs/interleaved/video set the VLM capability ceiling?
3. What are OBELICS/MMC4/WebVid-10M? What unique ability does interleaved-document pretraining bring?
4. How many vision tokens does a 336×336 image produce at patch=14? How do video frames scale the budget?
5. Interleaved vs paired data: which matters more for multi-image reasoning and long context, and why?

<details>
<summary>中文题目</summary>

1. VLM 预训练数据的三级金字塔是什么?每一级的特点与作用?
2. 图文对、交错文档、视频的配比如何影响 VLM 能力上限?
3. OBELICS / MMC4 / WebVid-10M 各是什么数据?交错文档带来什么独特能力?
4. 一张 336×336 图像在 ViT patch=14 下产生多少视觉 token?视频帧如何扩展?
5. 交错数据 vs 图文对数据: 哪个对多图推理与长上下文更关键?为什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/interleaved-data?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=interleaved-data)

---

## 15. JEPA Joint-Embedding Prediction · JEPA 联合嵌入预测

> JEPA (Joint Embedding Predictive Architecture) does predictive self-supervised learning in latent space: a context encoder $g$ takes the current latent $z_t$ and action $a_t$ and predicts the target representation $f(z_{t+1})$, where the target encoder $f$ is…

**Interview questions**

1. Write JEPA's latent prediction loss and explain $g$, $f$, $z_t$, $a_t$?
2. What does the EMA target encoder do? Why doesn't JEPA collapse without negative samples?
3. What is the essential difference between JEPA and VAE/diffusion reconstruction?
4. How does V-JEPA apply JEPA to video understanding? What representation is predicted?
5. JEPA vs contrastive learning (CLIP/InfoNCE): differences in objective and negative-sample usage?

<details>
<summary>中文题目</summary>

1. 写出 JEPA 的潜空间预测损失函数, 并解释 $g$、$f$、$z_t$、$a_t$ 的含义?
2. EMA 目标编码器起什么作用? 为什么 JEPA 不用负样本也不会表示塌缩?
3. JEPA 与 VAE/扩散重建式自监督方法的本质区别是什么?
4. V-JEPA 如何把 JEPA 用于视频理解? 预测的是什么表示?
5. JEPA 与对比学习 (如 CLIP/InfoNCE) 在训练目标与负样本使用上的异同?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/jepa?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=jepa)

---

## 16. Latent Diffusion (LDM) · 潜空间扩散 (LDM)

> Latent Diffusion Models (LDM) run the DDPM process in a VAE latent space instead of pixel space, then decode back. An image $x \in \mathbb{R}^{H \times W \times 3}$ is compressed by the VAE encoder into $z \in \mathbb{R}^{h \times w \times c}$;

**Interview questions**

1. Why does Stable Diffusion diffuse in VAE latent space instead of pixel space; how much compute does 8× downsampling save?
2. Which terms make up the LDM training loss, and why add perceptual and adversarial losses?
3. How is text conditioning injected into the latent diffusion network; where does cross-attention sit in U-Net/DiT?
4. Key structural differences between SD1/2 U-Net and SD3/Flux MMDiT?
5. Tradeoffs of latent vs pixel-space diffusion in training speed, memory and image quality?

<details>
<summary>中文题目</summary>

1. 为什么 Stable Diffusion 在 VAE 潜空间而不是像素空间扩散? 8× 降维省多少算力?
2. LDM 的总体训练损失由哪几部分组成? 为什么加感知损失和对抗项?
3. 文本条件如何注入潜空间扩散网络? 交叉注意力在 U-Net/DiT 中的位置?
4. SD1/2 的 U-Net 与 SD3/Flux 的 MMDiT 在结构上的关键差异?
5. 潜空间扩散相比像素空间扩散在训练速度、显存、图像质量上的权衡?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/latent-diffusion?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=latent-diffusion)

---

## 17. LLM Backbone (cross-module) · LLM 底座 (跨模块)

> VLMs and speech LLMs reuse the LLM as a unified backbone — Transformer architecture, attention, the pretrain-align-SFT paradigm, KV cache, and inference optimization all carry over from the LLM stack.

**Interview questions**

1. What does a VLM reuse and add on top of an LLM in architecture/training?
2. How are multimodal and text tokens unified in one Transformer backbone?
3. Why is a multimodal model's capability ceiling set by its LLM backbone?
4. KV-cache and inference-optimization challenges under long multimodal contexts (video/high-res)?
5. How does multimodal pretraining differ from LLM pretraining in alignment objectives?

<details>
<summary>中文题目</summary>

1. VLM 与 LLM 在架构/训练范式上分别复用与新增了什么?
2. 多模态 token 与文本 token 如何在同一个 Transformer 主干中统一处理?
3. 为什么说多模态模型的能力上限受限于 LLM 底座?
4. 长上下文 (视频/高分辨率图像) 下 KV Cache 与推理优化面临什么挑战?
5. 多模态预训练与 LLM 预训练在对齐目标上有何不同?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/llm-backbone?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=llm-backbone)

---

## 18. Masked Cross-Entropy Loss · Masked Cross-Entropy

> Masked Cross-Entropy (Masked CE) is the standard VLM training loss — sequence-level next-token prediction that computes loss only on assistant (model-answer) text tokens, masking gradients of user-instruction tokens and all vision patch tokens: $$\mathcal{L} =…

**Interview questions**

1. Write the masked-CE formula; which tokens are masked and why?
2. What goes wrong if you compute CE on vision patch tokens?
3. With vision tokens at 85–90% of the sequence, how does masking affect compute (336×336→576 patches)?
4. How does masked CE correspond to the ignore_index=-100 mechanism of LLM SFT?
5. Why mask user instructions too? What would the model learn otherwise?

<details>
<summary>中文题目</summary>

1. 写出 Masked CE 公式, 哪些 token 被屏蔽?为什么?
2. 对视觉 patch token 算 CE 会发生什么问题?
3. 视觉 token 占序列 85–90% 时, 屏蔽如何影响计算量 (336×336→576 patch)?
4. Masked CE 与 LLM SFT 的 ignore_index=-100 机制如何对应?
5. 为什么用户指令也要屏蔽?不屏蔽会学到什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/masked-cross-entropy?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=masked-cross-entropy)

---

## 19. Multimodal DPO / RLHF-V · 多模态偏好对齐

> Multimodal DPO (MMDPO) extends DPO from text-only preference alignment to joint (image, text) inputs: under the same image $v$, a faithful answer $y_w$ (consistent with the image) vs a hallucinated answer $y_l$ (contradicting it) forms a preference pair, optim…

**Interview questions**

1. Write the multimodal DPO loss; how does it differ from text DPO?
2. Why does DPO need no reward model or online sampling? Derive the implicit reward?
3. How does POPE sampling construct preference pairs (random/popular/adversarial)?
4. Relation and differences between MMDPO and RLHF-V?
5. Multimodal alignment: DPO vs RLHF/GRPO — scenario comparison?

<details>
<summary>中文题目</summary>

1. 写出多模态 DPO 的损失公式, 与文本 DPO 的差异在哪?
2. DPO 为什么不需要奖励模型和在线采样?隐式奖励如何推导?
3. POPE 负采样如何为 DPO 构造偏好对 (random/popular/adversarial)?
4. MMDPO 与 RLHF-V 的关系与区别?
5. 多模态对齐选 DPO 还是 RLHF/GRPO?适用场景对比?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/multimodal-dpo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multimodal-dpo)

---

## 20. Multimodal Eval Benchmarks · 多模态评测基准

> Multimodal benchmarks are specialized by dimension: MMBench for general ability, MMMU for academic reasoning, POPE for hallucination — each with its own question format and metrics.

**Interview questions**

1. Why is MMBench's ability taxonomy and single-letter MCQ design effective?
2. MMMU vs generic VQA benchmarks; why is visual information designed to be necessary?
3. POPE's three negative-construction strategies (random/popular/adversarial)?
4. Write POPE's Precision/Recall/F1 and yes-ratio; what does each measure?
5. Methodological differences among CHAIR / MMHal-Bench / HallusionBench?

<details>
<summary>中文题目</summary>

1. MMBench 的能力分类与单字母选择题设计为什么有效?
2. MMMU 与一般 VQA 基准的区别?为什么强调视觉信息是 necessary?
3. POPE 三种负样本构造策略 (random/popular/adversarial) 的区别?
4. 写出 POPE 的 Precision/Recall/F1 与 yes-ratio, 各自衡量什么?
5. CHAIR / MMHal-Bench / HallusionBench 的幻觉评测方法学差异?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/multimodal-evaluation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multimodal-evaluation)

---

## 21. Multimodal Hallucination · 多模态幻觉

> Multimodal Hallucination is VLM output that contradicts the image, mainly two kinds: object hallucination (mentioning objects not present) and attribute hallucination (wrong color/count/position).

**Interview questions**

1. Explain the three causes of object hallucination (text prior, co-occurrence bias, training objective)?
2. Write the VCD formula and explain why it mitigates hallucination?
3. How does POPE construct positive/negative probes? How is CHAIR computed?
4. Data-level hallucination mitigation: co-occurrence bias and hard negatives?
5. How do greedy/sampling decoding amplify hallucination? OPERA mechanism?

<details>
<summary>中文题目</summary>

1. 物体幻觉的三大成因 (文本先验/共现偏置/训练目标) 逐一解释?
2. 写出 VCD 视觉对比解码公式并解释为什么能缓解幻觉?
3. POPE 评估如何构造正负样本?CHAIR 指标怎么算?
4. 数据层面如何缓解物体幻觉 (共现偏置与硬负例)?
5. 贪心/采样解码如何放大幻觉?OPERA 的原理?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/multimodal-hallucination?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multimodal-hallucination)

---

## 22. Multimodal RAG (cross-module) · 多模态 RAG (跨模块)

> multimodal RAG uses CLIP-style multimodal embeddings to unify images and text in one vector space for joint retrieval, followed by reranking (cross-encoder) and generation.

**Interview questions**

1. How does multimodal RAG unify images and text into one vector space for joint retrieval?
2. Multimodal embeddings (e.g. CLIP) vs text-only embeddings for retrieval?
3. How to design recall/rerank pipelines for mixed image-text content?
4. Where does multimodal RAG beat text-only RAG the most?
5. Hard cases for table/chart document retrieval and common solutions?

<details>
<summary>中文题目</summary>

1. 多模态 RAG 如何把图文统一到同一向量空间做联合检索?
2. 多模态 embedding (如 CLIP) 与纯文本 embedding 在检索上的差异?
3. 图文混合内容检索的召回/重排管线如何设计?
4. 多模态 RAG 相比纯文本 RAG 在哪些场景收益最大?
5. 表格/图表类文档检索的难点与常用解法?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/rag-multimodal?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rag-multimodal)

---

## 23. Multimodal Reasoning & Agents · 多模态推理与 Agent

> Multimodal Reasoning is the multi-step reasoning ability of VLMs on chart reasoning (ChartQA: reading axes, extracting values, tracking trends), spatial reasoning (direction/position/size/count), mathematics (MathVista) and scientific figures (ScienceQA).

**Interview questions**

1. Multimodal CoT vs text CoT: differences and the core difficulty of chart reasoning?
2. VLM as agent perception: single-image, multi-image, and video-stream inputs?
3. Typical VLM failure modes on MathVista/ChartQA (numeric misreading, evidence decoupling)?
4. Why is spatial reasoning hard for VLMs and how is it improved?
5. Does multimodal reasoning come mainly from pretraining, SFT, or decoding? Evidence?

<details>
<summary>中文题目</summary>

1. 多模态 CoT 与纯文本 CoT 的差异?图表推理的核心难点?
2. VLM 作为 Agent 感知层的三种输入形态 (单图/多图/视频流) 及各自处理?
3. MathVista / ChartQA 上 VLM 常见的错误类型 (数值读取/证据脱节)?
4. 空间推理为什么难?如何改进?
5. 多模态推理能力主要来自预训练、SFT 还是解码策略?证据是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/multimodal-reasoning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multimodal-reasoning)

---

## 24. Native Any-to-Any Multimodal · Native 同构多模态

> Native (homogeneous) multimodal models unify images, audio and text into one discrete token space and run autoregressive next-token prediction on a single Transformer backbone: Chameleon (Meta) quantizes images into discrete VQGAN image tokens mixed with text…

**Interview questions**

1. Native multimodal (GPT-4o/Chameleon) vs modular VLM (LLaVA): core differences?
2. How are images discretized into tokens for autoregressive training? Role of VQGAN?
3. Why is GPT-4o voice low-latency? Benefits of audio tokenization?
4. Cost of a unified token space? Why may text ability get diluted?
5. How do native models handle video/long context? Token-count and compute challenges?

<details>
<summary>中文题目</summary>

1. 原生多模态 (GPT-4o/Chameleon) 与模块化 VLM (LLaVA) 的核心区别?
2. 图像怎么离散成 token 参与自回归训练?VQGAN 的作用?
3. GPT-4o 语音对话为什么延迟低?音频 token 化的好处?
4. 统一 token 空间的代价是什么?为什么文本能力可能被稀释?
5. native 架构如何处理视频/长上下文?token 数与算力挑战?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/native-multimodal?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=native-multimodal)

---

## 25. Progressive Video Training · 视频渐进训练

> Text-to-video training follows a three-stage progressive recipe (Sora, Stable Video Diffusion, Kling and others), gradually introducing the video distribution: 1.

**Interview questions**

1. What does each of the three training stages learn, and why is the order fixed?
2. What does aspect-ratio bucketing solve, and how is a single batch organized?
3. Why filter training clips with optical flow, and how is the threshold set?
4. Learning-rate and data-selection choices in the HD fine-tune stage?
5. Why does image pretraining benefit video generation?

<details>
<summary>中文题目</summary>

1. 视频训练三阶段各自训什么?为什么顺序不能反?
2. 宽高比分桶解决什么问题?一个 batch 内的样本如何组织?
3. 为什么要用光流过滤训练数据?阈值如何设定?
4. 高清微调阶段如何设置学习率与数据选择?
5. 图像预训练为什么对视频生成有帮助?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/video-training-pipeline?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=video-training-pipeline)

---

## 26. RVQ Codebooks & Tokens · RVQ 码本

> RVQ (Residual Vector Quantization) cascades $K$ codebooks, each quantizing the residual left by earlier levels — the core quantizer of neural audio codecs (SoundStream / EnCodec / DAC) and speech LLMs (Vall-E, CosyVoice).

**Interview questions**

1. Write RVQ's residual-quantization formula; what is quantized at level k, and the reconstruction?
2. RVQ vs single-stage VQ; why does cascading K levels balance bitrate and quality?
3. How are RVQ codebooks trained; commitment loss and EMA updates?
4. What do semantic vs acoustic tokens encode, and from which levels?
5. Why predict semantic before acoustic tokens; AR vs NAR decoding roles?

<details>
<summary>中文题目</summary>

1. 写出 RVQ 残差量化公式:第 k 级量化的是什么?重建公式是什么?
2. RVQ 与单级 VQ 的区别?为什么级联 K 级能兼顾码率与音质?
3. RVQ 码本如何训练?commitment loss 与 EMA 码本更新?
4. Semantic token 与 Acoustic token 分别编码什么?各自来自哪些层级?
5. 语音 LLM 为什么先预测 semantic 再预测 acoustic?各用什么解码方式?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/rvq-codebook?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rvq-codebook)

---

## 27. Sampler Acceleration (DDIM/DPM-Solver/LCM) · 采样加速 (DDIM/DPM-Solver/LCM)

> Original DDPM sampling needs 1000 steps; acceleration falls into three families: 1. DDIM (Denoising Diffusion Implicit Models, deterministic implicit sampling): rewrites the forward chain as non-Markovian so sampling becomes a deterministic trajectory of the p…

**Interview questions**

1. Write the DDIM sampling formula; why is σ_t=0 a deterministic ODE sampler?
2. Do DDIM and DDPM share the same training objective; why does DDIM need no retraining?
3. How do LCM/consistency models achieve 4-step generation; what does the consistency loss constrain?
4. Difference between DPM-Solver/DPM++ and DDIM; why do high-order solvers need fewer steps for better quality?
5. Steps vs FID/detail tradeoff; when to use 4-step LCM vs 20-step DPM++?

<details>
<summary>中文题目</summary>

1. 写出 DDIM 采样公式, σ_t=0 时为什么是确定性 ODE 采样?
2. DDIM 与 DDPM 训练目标是否相同? 为什么 DDIM 无需重训练?
3. LCM/一致性模型如何做到 4 步出图? 一致性损失约束的是什么?
4. DPM-Solver/DPM++ 与 DDIM 的区别? 为什么高阶求解器步数更少质量更好?
5. 采样步数与 FID/细节的权衡? 4 步 LCM vs 20 步 DPM++ 各自适用场景?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/sampler-acceleration?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=sampler-acceleration)

---

## 28. Speech-Audio LLM · 语音大模型

> Speech-audio LLMs let a large language model “hear” audio directly (audio in, sometimes audio out); the core question is how audio is fused into the LLM, with two dominant designs: 1.

**Interview questions**

1. The two main ways to fuse audio into LLMs, with pros and costs?
2. Qwen-Audio architecture: how encoder + Q-Former compress audio into tokens?
3. Why is GPT-4o's native audio-token design low-latency vs cascaded STT/TTS?
4. Audio tokens are far denser than text — how is the mismatch handled?
5. Speech-LLM staged training: why freeze the LLM while aligning the encoder?

<details>
<summary>中文题目</summary>

1. 语音接入 LLM 的两种主流方式?各自优点与代价?
2. Qwen-Audio 架构细节:编码器 + Q-Former 如何把音频压缩成 token?
3. GPT-4o 原生音频 token 方案为什么延迟低?与级联 STT/TTS 相比?
4. 音频 token 密度远高于文本,如何解决?
5. 语音 LLM 分阶段训练:为什么阶段 1 冻结 LLM 只对齐音频编码器?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/speech-audio-llm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=speech-audio-llm)

---

## 29. TTS · 语音合成 TTS

> TTS (text-to-speech) generates speech from text. Pipeline evolution: concatenative/parametric synthesis (HMM, vocoders like Griffin-Lim phase recovery) → two-stage neural TTS (Tacotron 2 acoustic model + WaveGlow/HiFi-GAN vocoder) → end-to-end (VITS, Vall-E, C…

**Interview questions**

1. TTS evolution: concatenative → two-stage → end-to-end; the bottleneck of each?
2. Why variational inference in VITS; write the ELBO and explain each term?
3. The role of normalizing flow in VITS; why not a plain Gaussian prior?
4. What does Monotonic Alignment Search solve, and why no external aligner?
5. VITS vs two-stage TTS; what does the GAN discriminator contribute?

<details>
<summary>中文题目</summary>

1. TTS 管线演进:拼接 → 两阶段 → 端到端,各自瓶颈是什么?
2. VITS 为什么用变分推理?写出训练目标 ELBO 并解释每一项?
3. normalizing flow 在 VITS 中扮演什么角色?为什么不用普通高斯先验?
4. 单调对齐搜索(MAS)解决什么问题?为什么不需要外部对齐器?
5. VITS 与两阶段 TTS 的对比?GAN 判别器贡献了什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/tts?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=tts)

---

## 30. Video Generation · 视频生成

> Video generation adds temporal modeling on top of image diffusion, treating a video as a high-dimensional tensor $x \in \mathbb{R}^{F \times H \times W \times 3}$ ($F$ RGB frames).

**Interview questions**

1. How do video diffusion models model time — tradeoffs among 3D conv, factorized attention, and causal attention?
2. How is the causal attention mask written, and what does it unlock?
3. What is cascade generation, and what does each spatio-temporal upsampler do?
4. Common video artifacts (flicker/drift/repetition) — causes and mitigations?
5. Key video training-data challenges compared with image data?

<details>
<summary>中文题目</summary>

1. 视频扩散如何建模时间维?3D 卷积 / 时空分离注意力 / 因果注意力三种方式的取舍?
2. 因果注意力 mask 怎么写?它解锁了哪些生成能力(任意时长/分辨率)?
3. 什么是级联生成?两级时空超分各自负责什么?
4. 视频生成常见伪影(闪烁/漂移/重复)的原因与缓解手段?
5. 视频训练数据的主要挑战?与图像数据相比缺什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/video-generation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=video-generation)

---

## 31. Vision Encoder · 视觉编码器

> The vision encoder of a VLM usually reuses CLIP's ViT (CLIP-ViT-L/14 is the LLaVA-1.5 default) or SigLIP (the LLaVA-1.6 upgrade): the image is split into $p \times p$ patches, position-encoded, and passed through a Transformer, with token count…

**Interview questions**

1. How many visual tokens does 336×336 + ViT-14 produce? Give the formula.
2. Why freeze the vision encoder in early VLM stages? When is it unfrozen?
3. CLIP ViT vs SigLIP: what does the sigmoid loss fix?
4. Is a larger vision ViT always better? How to weigh training cost vs gain?
5. With a frozen vision tower, why can high-resolution ability still emerge (AnyRes patching)?

<details>
<summary>中文题目</summary>

1. 336×336 输入 + ViT-14 patch 会产生多少个 visual token?写出公式?
2. 为什么 VLM 前几阶段要冻结视觉编码器?什么时候解冻?
3. CLIP ViT 与 SigLIP 的区别?sigmoid loss 解决了什么问题?
4. 视觉塔选更大的 ViT 一定更好吗?训练成本与收益怎么权衡?
5. 冻结视觉塔时为什么仍能学到高分辨率能力 (AnyRes 切块原理)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/vision-encoder?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vision-encoder)

---

## 32. VLA Models (cross-module) · VLA 具身大模型 (跨模块)

> VLA (Vision-Language-Action) models unify vision, language, and action in one autoregressive generator. RT-2 discretizes each dimension of the 7-DOF end-effector pose (3D translation + 3D rotation + gripper) into 0–256 bin tokens appended to the vocabulary, ge…

**Interview questions**

1. How does RT-2 discretize 7-DOF continuous actions into 0–256 bin tokens? Write the quantization formula?
2. Action discretization vs Diffusion Policy: essential difference and what each solves?
3. OpenVLA vs RT-2: differences (backbone, training data, action output)?
4. Why can VLAs reuse VLM priors? How are action and text tokens unified in training?
5. Write Diffusion Policy's diffusion training objective; its relation to DDPM image denoising?

<details>
<summary>中文题目</summary>

1. RT-2 如何把 7-DOF 连续动作离散化为 0–256 的 bin token? 写出量化公式?
2. 动作离散化与 Diffusion Policy 生成动作的本质区别? 各解决什么问题?
3. OpenVLA 与 RT-2 的异同 (底座、训练数据、动作输出方式)?
4. VLA 为什么能复用 VLM 先验? 动作 token 与文本 token 如何统一训练?
5. 写出 Diffusion Policy 的扩散训练目标, 与 DDPM 图像去噪的关系?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/vla-embodied?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vla-embodied)

---

## 33. VLM 3-Stage Pipeline · VLM 三阶段训练

> The VLM 3-stage pipeline (LLaVA/Qwen2-VL paradigm) proceeds through alignment → interleaved pretraining → visual instruction SFT, with per-stage freezing: | Stage | Vision encoder | Projector | LLM | Data | Purpose | |:--|:--|:--|:--|:--|:--| | Stage 1 alignme…

**Interview questions**

1. Draw the 3-stage VLM pipeline table: what is frozen/trained and which data per stage?
2. Why is the vision encoder frozen throughout? Consequences of unfreezing it?
3. Purpose of Stage-1 projector-only training? Projector input/output dimensions?
4. Stage-3 full fine-tuning vs LoRA: when to use each and the criteria?
5. How does the 3-stage recipe map onto LLaVA's 2-stage and Qwen2-VL's 3-stage?

<details>
<summary>中文题目</summary>

1. 画出 VLM 三阶段训练流程表: 每阶段冻结什么、训练什么、用什么数据?
2. 为什么视觉编码器全程冻结?解冻会有什么后果?
3. Stage 1 只训投影器的目的?投影器的输入输出维度?
4. Stage 3 何时用全参微调、何时用 LoRA?选择依据?
5. 三阶段配方与 LLaVA 两阶段 / Qwen2-VL 三阶段的对应关系?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/vlm-three-stage?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vlm-three-stage)

---

## 34. VLM Architecture · VLM 架构

> The standard VLM (Vision-Language Model) recipe is three parts — vision encoder + projector + LLM: the vision encoder (CLIP ViT) patchifies the image into visual features $z_{vis}$, the projector (MLP/Q-Former) maps them into the LLM's text embedding space, an…

**Interview questions**

1. The VLM three-part recipe: responsibilities of encoder / projector / LLM?
2. LLaVA (input-level fusion) vs Flamingo (layer-level cross-attention): differences?
3. Why project visual features into the text embedding space? How is the dimension mismatch handled?
4. Why is the VLM loss cross-entropy over text tokens only? Is there no loss on images?
5. Native multimodal (GPT-4o/Chameleon) vs modular VLM (LLaVA): architecture trade-offs?

<details>
<summary>中文题目</summary>

1. VLM 三段式架构 (编码器/投影器/LLM) 各自的职责是什么?
2. LLaVA (输入层融合) 与 Flamingo (层内 cross-attention) 的区别?
3. 为什么视觉特征要投影到文本嵌入空间?维度不匹配怎么处理?
4. VLM 训练目标为什么只算文本 token 的交叉熵?图像部分没有 loss 吗?
5. 原生多模态 (GPT-4o/Chameleon) 与模块化 VLM (LLaVA) 的架构取舍?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/vlm-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vlm-architecture)

---

## 35. VLM Capabilities · VLM 能力图谱

> VLM Capabilities is the unified set of autoregressive abilities of vision-language models: image captioning, visual QA (VQA), OCR (TextVQA/DocVQA), visual grounding (referring localization), multi-image comparison, chart and video understanding.

**Interview questions**

1. Why do caption/VQA/OCR/grounding all reduce to autoregressive token generation? Write the unified objective.
2. How is visual grounding implemented? How are continuous boxes discretized into tokens?
3. Text hallucination vs OCR hallucination: difference, causes and examples?
4. VLM capability boundaries: what data and input representations enable OCR/chart/video?
5. Multi-image comparison vs video understanding: differences in input representation?

<details>
<summary>中文题目</summary>

1. 为什么 VLM 的 caption/VQA/OCR/grounding 都归结为自回归 token 生成?写出统一目标?
2. 视觉 grounding 如何实现?连续坐标框如何离散化为 token?
3. 文本幻觉与 OCR 幻觉的区别?各自的成因与例子?
4. VLM 能力边界: OCR / 图表 / 视频各需要什么数据与输入表示支撑?
5. 多图比较与视频理解在输入表示上的差异?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/vlm-capabilities?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vlm-capabilities)

---

## 36. VLM Projector MLP/Q-Former · VLM 投影器 MLP/Q-Former

> The projector maps visual features into the LLM's text embedding space; three mainstream routes: ① LLaVA's linear/two-layer MLP — $h = \text{MLP}(z_{vis})$ maps each visual token independently (typically 1024→4096 dims), preserving token count and spatial stru…

**Interview questions**

1. Write the MLP projection formula $h = \text{MLP}(z_{vis})$; how do token count and dims change in LLaVA?
2. How does Q-Former compress 576 visual tokens into 32? Role of learnable queries?
3. MLP projection vs Q-Former compression: trade-offs? Which tasks favor which?
4. How is the projector trained in the alignment stage? Is it updated together with the LLM?
5. How does C-Abstractor downsample? How many tokens does it save?

<details>
<summary>中文题目</summary>

1. 写出 MLP 投影公式 $h = \text{MLP}(z_{vis})$; LLaVA 投影前后 token 数与维度怎么变?
2. Q-Former 怎么把 576 个视觉 token 压缩成 32 个?可学习 query 的作用?
3. MLP 投影与 Q-Former 压缩的取舍?什么任务适合哪种?
4. 投影器在预对齐阶段怎么训练?与 LLM 是否同步更新?
5. C-Abstractor 降采样的原理?token 数减少多少?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/vlm-projector?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vlm-projector)

---

## 37. World Model Applications · 世界模型应用

> world-model applications use the learned environment-dynamics prior for three tasks — video prediction/understanding: V-JEPA predicts future representations in latent space without reconstructing pixels;

**Interview questions**

1. How is a world model used for video prediction, video generation, and planning? One representative model each?
2. V-JEPA vs Sora: essential difference (predict representations vs generate pixels)?
3. Why do imagined rollouts improve RL sample efficiency? How does model error affect planning quality?
4. What does Sora's 3D VAE spatio-temporal compression do? Compression ratios in space and time?
5. Practical difficulties of using generative video world models directly for planning?

<details>
<summary>中文题目</summary>

1. 世界模型在视频预测、视频生成、规划三条路线分别怎么用? 各举一个代表模型?
2. V-JEPA 与 Sora 的本质区别 (只预测表示 vs 生成像素)?
3. 想象 rollout 为什么能提升 RL 样本效率? 模型误差如何影响规划质量?
4. Sora 的 3D VAE 时空压缩做什么? 空间/时间各压缩多少倍?
5. 生成式视频世界模型直接用于规划有哪些落地难点?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Multimodal/card/world-model-application?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=world-model-application)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
