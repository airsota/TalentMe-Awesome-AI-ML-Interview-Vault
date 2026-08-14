<div align="center">

# 🌟 TalentMe: Awesome AI / ML / LLM Technical Vault & Interview Mastery

**The Ultimate Bilingual (English / 中文) Open-Source Technical Knowledge Base, Mathematical Derivation Codex, and Role-Driven Interview Preparation Architecture.**

[![GitHub Stars](https://img.shields.io/github/stars/airsota/TalentMe-Awesome-AI-ML-Interview-Vault?style=for-the-badge&logo=github&color=8b5cf6)](https://github.com/airsota/TalentMe-Awesome-AI-ML-Interview-Vault)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Bilingual](https://img.shields.io/badge/Language-English%20%7C%20%E4%B8%AD%E6%96%87-10b981?style=for-the-badge)](README.md)
[![Live Platform](https://img.shields.io/badge/Interactive%20Web-TalentMe%20Live-ec4899?style=for-the-badge&logo=firefox)](https://talentme.airsota.com/resources/tech)
[![Mindmaps](https://img.shields.io/badge/Roadmap-13%20Interactive%20Mindmaps-3b82f6?style=for-the-badge)](https://talentme.airsota.com/resources/tech/roadmap)

<br/>

<p align="center">
  <a href="https://talentme.airsota.com/resources/tech"><b>🌐 Live Interactive Vault</b></a> •
  <a href="https://talentme.airsota.com/resources/tech/roadmap"><b>🧭 Interactive Mindmaps</b></a> •
  <a href="#-4-specialized-role-fast-tracks"><b>🎯 4 Role Tracks</b></a> •
  <a href="#-13-foundational-modules-overview"><b>📖 13 Modules</b></a> •
  <a href="#-pure-python-industrial-operator-showcase"><b>🛠️ Python Operators</b></a>
</p>

</div>

---

## 🚀 Overview & Philosophy

**TalentMe Awesome AI/ML Interview Vault** is engineered to bridge the canyon between academic mathematical theory and 10,000-GPU industrial deployment. 

Unlike generic interview prep lists with superficial answers, this repository provides:
1. **Exhaustive Deep Dives**: 70+ bilingual in-depth guides with rigorous algebra, closed-form derivations, and zero hand-waving.
2. **477 Atomic Knowledge Cards**: Core exam points, failure modes, and interviewer follow-up questions.
3. **Pure Python Implementations**: Standalone, zero-heavy-dependency Python / NumPy operators with unit test cases.
4. **Dual Perspective**: Learn bottom-up via **Foundational Domains** or top-down via **Target Role Tracks** (`MLE`, `AIE`, `RS`, `DS`).

---

## 🎯 4 Specialized Role Fast-Tracks

| Role Track | Focus Areas | Key Mathematical Derivations & System Design | Direct Access |
|---|---|---|:---:|
| 💻 **MLE**<br/>*(Machine Learning Engineer)* | Recommendation funnels, Search/Ads, Feature Stores, Live-coding | • DSSM Two-Tower with $\log(p_j)$ In-batch debiasing<br/>• MMoE Multi-gate Mixture-of-Experts<br/>• ESMM Entire Space Multi-Task Modeling (eliminating SSB)<br/>• DPP (Determinantal Point Processes) diversity greedy search | [📂 MLE Guides](Interviews/MLE/) |
| 🚀 **AIE**<br/>*(AI Systems Engineer)* | Production RAG, Agent systems, Fine-tuning, vLLM serving | • SFT Data Packing algorithms & Loss Masking (`labels=-100`)<br/>• LoRA $B=0$ initialization & QLoRA NF4 VRAM Calculus<br/>• BM25 + Dense RRF ($k=60$) reciprocal rank fusion<br/>• Speculative Decoding memory-bound acceptance sampling | [📂 AIE Guides](Interviews/AIE/) |
| 🎓 **RS**<br/>*(Research Scientist)* | SOTA Paper deconstruction, Alignment math, Scaling laws | • DPO implicit reward substitution & $Z(x)$ cancellation<br/>• PPO pessimistic clipped lower-bound proof<br/>• RoPE complex inner product relative displacement isomorphism<br/>• DeepSeek-R1 pure RL emergence & inference scaling laws | [📂 RS Guides](Interviews/RS/) |
| 📈 **DS**<br/>*(Data Scientist)* | Causal inference, Advanced A/B testing, Uplift modeling | • Rubin Potential Outcomes & Pearl Causal DAG $d$-separation<br/>• SCM (Synthetic Control) convex weights & placebo permutation<br/>• CUPED algebraic variance reduction proof<br/>• SRM Chi-square $\chi^2$ goodness-of-fit diagnostic funnel | [📂 DS Guides](Interviews/DS/) |

---

## 📖 13 Full-Spectrum Foundational Modules & System Topology

```mermaid
graph TD
    %% ==========================================
    %% Layer 1: Mathematical Foundations & ML
    %% ==========================================
    subgraph L1["📐 Layer 1: Mathematical Foundations & Classic Learning"]
        M1["Math & Optimization<br/>• SVD & Spectral PCA<br/>• Matrix Calculus & Hessians<br/>• KKT & Lagrangian Duals<br/>• Itô Stochastic Calculus"]
        M2["Classic Machine Learning<br/>• Logistic Regression<br/>• SVM Max-Margin & Duals<br/>• GBDT & XGBoost 2nd-Order<br/>• LightGBM GOSS & EFB"]
        M1 --> M2
    end

    %% ==========================================
    %% Layer 2: Deep Learning & Frontier Models
    %% ==========================================
    subgraph L2["🧠 Layer 2: Deep Learning & Frontier Architectures"]
        M3["Deep Learning Foundations<br/>• ResNet & ViT Projection<br/>• RNN / LSTM / Mamba SSM<br/>• Kaiming Init & AdamW<br/>• RMSNorm & WGAN-GP"]
        M4["Large Language Models (LLMs)<br/>• Multi-Head Attention & RoPE<br/>• KV Cache (MLA / GQA)<br/>• SFT Loss Masking & LoRA<br/>• MoE Routing & Load Balancing"]
        M5["RL & Post-Training Alignment<br/>• MDP & Bellman Optimality<br/>• PPO Pessimistic Clipped Bound<br/>• DPO Implicit Reward Substitution<br/>• GRPO Pure RL Emergence"]
        M6["Multimodal & Diffusion<br/>• CLIP Contrastive Learning<br/>• VLM Visual Projectors<br/>• DDPM SDE & Flow Matching<br/>• DiT Scaling Architectures"]
        
        M3 --> M4
        M4 --> M5
        M4 --> M6
    end

    %% ==========================================
    %% Layer 3: Infrastructure & System Design
    %% ==========================================
    subgraph L3["🖥️ Layer 3: Infrastructure, Compound Systems & Industrial Design"]
        M7["AI Infrastructure & Clusters<br/>• ZeRO-1/2/3 16Ψ Partitioning<br/>• Megatron 3D Parallelism<br/>• FlashAttention Online Softmax<br/>• vLLM PagedAttention"]
        M8["AI Engineering & Agents<br/>• BM25 + Dense RRF (k=60)<br/>• Cross-Encoder Reranking<br/>• Agent State Machines & Loops<br/>• SWE-bench Sandboxed Eval"]
        M9["Industrial System Design<br/>• 4-Stage Funnel (10M→10)<br/>• Real-time Feature Stores<br/>• Point-in-Time Joins<br/>• Risk Control & Fraud GNN"]
        
        M7 --> M8
        M8 --> M9
    end

    %% ==========================================
    %% Layer 4: 4 Specialized Role Fast-Tracks
    %% ==========================================
    subgraph L4["🎯 Layer 4: 4 Specialized Role Fast-Tracks"]
        R1["💻 MLE Fast-Track<br/>DSSM log(p) Debiasing → MMoE / DCN-v2 → DPP Diversity Selection"]
        R2["🚀 AIE Fast-Track<br/>SFT Data Packing → LoRA VRAM Calculus → Enterprise GraphRAG"]
        R3["🎓 RS Fast-Track<br/>Paper 4-Step Deconstruction → DPO/PPO Proofs → Scaling Laws"]
        R4["📈 DS Fast-Track<br/>Rubin DAG d-Sep → SCM/IV → CUPED Variance Reduction → SRM Funnel"]
    end

    %% Cross-Layer Connectors
    L1 --> L2
    L2 --> L3
    L3 --> L4
```

### 1. [📐 Math & Optimization (Foundations/Math)](Foundations/Math/)
* **`math-linear-algebra-and-matrix-calculus`** (`.zh.md` / `.en.md`): SVD, PCA spectral decomposition, Jacobians, Hessians, and matrix calculus layout conventions.
* **`math-probability-and-information-theory`** (`.zh.md` / `.en.md`): MLE, MAP, KL divergence, Cross-Entropy, and Monte Carlo Markov Chains (MCMC).
* **`math-convex-optimization-and-calculus`** (`.zh.md` / `.en.md`): KKT conditions, Lagrangian duals, SGD convergence rates, and stochastic calculus (Itô's Lemma).

### 2. [📊 Classic Machine Learning (Foundations/ML)](Foundations/ML/)
* **`classic-ml-cheatsheet`** (`.zh.md` / `.en.md`): Linear/Logistic regression, SVM max-margin derivations, Kernel trick, Bias-Variance tradeoff.
* **`tree-models-and-ensemble-learning`** (`.zh.md` / `.en.md`): Random Forest, GBDT, XGBoost Taylor second-order expansion, LightGBM GOSS & EFB.
* **`clustering-and-dim-reduction`** (`.zh.md` / `.en.md`): K-Means++, GMM EM derivation, t-SNE perplexity, UMAP fuzzy simplicial sets.

### 3. [🧠 Deep Learning Foundations (Foundations/DL)](Foundations/DL/)
* **`cnn-and-vit-architectures`** (`.zh.md` / `.en.md`): ResNet residual skip connections, ConvNeXt, Vision Transformer (ViT) patch projection.
* **`rnn-lstm-and-mamba-ssm`** (`.zh.md` / `.en.md`): LSTM gates, GRU, State Space Models (SSM), S4, Mamba selective scan mechanism.
* **`optimizer-and-initialization`** (`.zh.md` / `.en.md`): Xavier / He (Kaiming) initialization variance proof, Adam, AdamW decoupled weight decay, Muon optimizer.
* **`normalization-and-regularization`** (`.zh.md` / `.en.md`): BatchNorm, LayerNorm, RMSNorm, Weight Decay vs $L_2$ regularization, Dropout.
* **`gan-and-generative-basics`** (`.zh.md` / `.en.md`): Minimax game, JS divergence collapse, WGAN-GP Wasserstein-1 distance & Kantorovich-Rubinstein duality.

### 4. [⚡ Large Language Models (Foundations/LLM)](Foundations/LLM/)
* **`transformer-architecture-core`** (`.zh.md` / `.en.md`): Multi-Head Attention, RoPE 2D complex rotation, SwiGLU, RMSNorm, Pre-LN vs Post-LN stability.
* **`kv-cache-and-decoding-mechanisms`** (`.zh.md` / `.en.md`): KV Cache memory calculus, MQA, GQA, MLA (Multi-Head Latent Attention), PagedAttention.
* **`llm-training-peft-and-moe`** (`.zh.md` / `.en.md`): SFT, LoRA, QLoRA, Mixture of Experts (MoE) Top-K routing & load balancing auxiliary loss.

### 5. [🎮 RL & Post-Training Alignment (Foundations/RL)](Foundations/RL/)
* **`rl-and-alignment-cheatsheet`** (`.zh.md` / `.en.md`): MDP, Bellman optimality, Policy Gradients, PPO, DPO, KTO, GRPO (DeepSeek-R1).

### 6. [👁️ Multimodal & Diffusion Models (Foundations/Multimodal)](Foundations/Multimodal/)
* **`clip-and-vision-language-models`** (`.zh.md` / `.en.md`): CLIP contrastive pretraining, LLaVA visual projection tokens, Q-Former.
* **`diffusion-models-and-dit`** (`.zh.md` / `.en.md`): DDPM forward/reverse SDE, DDIM deterministic sampling, Classifier-Free Guidance (CFG), Flow Matching, DiT.

### 7. [🖥️ AI Infrastructure & Distributed Training (Foundations/AI_Infra)](Foundations/AI_Infra/)
* **`distributed-training-and-parallelism`** (`.zh.md` / `.en.md`): Data Parallel (DDP), ZeRO-1/2/3 $16\Psi$ VRAM partitioning, Megatron 3D parallelism (TP/PP/DP/SP).
* **`gpu-architecture-and-kernel-optimization`** (`.zh.md` / `.en.md`): NVIDIA Hopper/Blackwell Tensor Cores, FlashAttention-1/2/3 online softmax tiling, Triton kernels.

### 8. [🤖 AI Engineering & Compound Systems (Foundations/AI_Engineering)](Foundations/AI_Engineering/)
* **`rag-advanced-systems`** (`.zh.md` / `.en.md`): Chunking strategies, HNSW vector indexing, BM25 + Vector RRF, Cross-Encoder reranking, GraphRAG.
* **`agentic-workflows-and-state-machines`** (`.zh.md` / `.en.md`): LangGraph / ReAct agent state machines, tool calling schemas, infinite loop guardrails, SWE-bench evaluations.

### 9. [🏗️ Industrial System Design (Foundations/System_Design)](Foundations/System_Design/)
* **`industrial-recsys-architecture`** (`.zh.md` / `.en.md`): 4-Stage RecSys funnel (Retrieval $\to$ Pre-Ranking $\to$ Heavy Ranking $\to$ Re-Ranking), Feature Store Point-in-Time Join.

---

## 🛠️ Pure Python Industrial Operator Showcase

All core mathematical algorithms in this repository are accompanied by self-contained, dependency-free Pure Python / NumPy implementations:

```python
# Determinantal Point Processes (DPP) Greedy Selection for RecSys Diversity
import numpy as np

def pure_python_dpp_greedy(quality_scores: np.ndarray, similarity_matrix: np.ndarray, max_items: int = 3) -> list[int]:
    n = len(quality_scores)
    selected = []
    L = np.outer(quality_scores, quality_scores) * similarity_matrix
    
    for _ in range(max_items):
        best_gain, best_idx = -1.0, -1
        for candidate in range(n):
            if candidate in selected: continue
            current_set = selected + [candidate]
            sub_L = L[np.ix_(current_set, current_set)]
            gain = np.linalg.det(sub_L)
            if gain > best_gain:
                best_gain, best_idx = gain, candidate
        if best_idx != -1 and best_gain > 0:
            selected.append(best_idx)
        else:
            break
    return selected
```

---

## 🧠 Native Obsidian Vault Support (1-Click Local Second Brain)

Every markdown document in this repository adheres strictly to the **Karpathy LLM-Wiki / Second Brain Specification**:
* **100% Native Markdown**: Compatible with [Obsidian](https://obsidian.md/), Logseq, and Foam.
* **Standard YAML Frontmatter**: Includes metadata tags, category classification, bilingual summaries, and FAQ schema.
* **KaTeX LaTeX Math Rendering**: All equations use standard `$...$` inline and `$$...$$` block syntax.
* **Interactive Mermaid Diagrams**: Directly visualizes workflows inside Obsidian's native renderer.

### Quick Setup with Obsidian:
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/airsota/TalentMe-Awesome-AI-ML-Interview-Vault.git
   ```
2. Open **Obsidian** $\to$ Click **"Open folder as vault"** $\to$ Select `TalentMe-Awesome-AI-ML-Interview-Vault`.
3. Press `Ctrl + G` (or `Cmd + G`) to open the **Interactive Knowledge Graph View** connecting all mathematical foundations and role-track guides!

---

## ⚡ TalentMe MCP: Supercharge Local Memory with AI Mock Interviewers & Cloud Sync

The true superpower of this repository is that **it is not just static text** — it seamlessly acts as your **Local Second Brain**, and when connected with the [**TalentMe MCP Server (GitHub: airsota/talentme-mcp)**](https://github.com/airsota/talentme-mcp), your AI assistant (Cursor, Claude Code, Windsurf, or Antigravity IDE) gains simultaneous read/write access to both your **Local Knowledge Vault** and the **TalentMe Cloud Knowledge Base**.

```mermaid
graph TD
    %% ==========================================
    %% 1. Local Second Brain & IDE
    %% ==========================================
    subgraph S1["🏠 1. Your Local Machine & Second Brain"]
        V1["🧠 Current Repository Vault<br/>• 157 In-Depth Guides<br/>• 477 Atomic Knowledge Cards<br/>• Personal Study Notes & Logs"]
        IDE1["💻 AI Coding IDE & Agent Terminal<br/>• Cursor AI / Claude Code<br/>• Google Antigravity / Windsurf"]
        V1 <--> IDE1
    end

    %% ==========================================
    %% 2. TalentMe MCP Protocol Server
    %% ==========================================
    subgraph S2["⚡ 2. TalentMe MCP Server (github.com/airsota/talentme-mcp)"]
        MCP_Core["Model Context Protocol Core Gateway<br/>Bi-directional Dual-Memory Context Engine"]
        
        T1["🎙️ tm-mock / mock-interview<br/>Terminal Live Interactive Stress Interviewer"]
        T2["📊 tm-assess & study-planner<br/>14-Day Baseline Diagnostics & Sprint Architect"]
        T3["🔄 bridge-sync-and-digest<br/>Cloud-to-Local Knowledge Distillation"]
        T4["📝 tm-debrief<br/>Post-Interview Feedback & Error Logging"]
        T5["🔁 tm-review<br/>Ebbinghaus Spaced Repetition Practice"]
        
        MCP_Core --> T1
        MCP_Core --> T2
        MCP_Core --> T3
        MCP_Core --> T4
        MCP_Core --> T5
    end

    %% ==========================================
    %% 3. TalentMe Cloud Knowledge Base
    %% ==========================================
    subgraph S3["🌐 3. TalentMe Master Cloud Knowledge Base"]
        Cloud_KB["TalentMe Cloud Master Graph<br/>• Real-Time Big Tech Interview Pipeline<br/>• SOTA Paper & System Architecture Feeds<br/>• Interactive Mindmap Topology Database"]
    end

    %% Connectors
    IDE1 <== "Model Context Protocol (JSON-RPC)" ==> MCP_Core
    T3 <== "Real-Time Cloud Query & Ingestion" ==> Cloud_KB
    T1 -. "Live Company Rubric Cross-Ref" .-> Cloud_KB
```

### 🌟 Why This Architecture Changes Everything:

1. **📁 100% Local Privacy & Ownership**:
   - The entire vault lives on your local disk. You can annotate, add your own company notes, and customize without vendor lock-in.
2. **☁️ Zero-Latency Cloud Knowledge Expansion**:
   - Through `talentme-mcp`, your IDE agent can pull the latest SOTA interview debriefs and architectural deep dives directly from TalentMe Cloud and distill them into your local vault.
3. **🎙️ Proactive Terminal Mock Interviews**:
   - Type `/mock` or invoke `tm-mock` in Cursor/Claude Code. The AI interviewer will examine you on topics in this repository (e.g. *DPO derivation*, *LoRA memory sizing*, *CUPED variance proofs*), probe edge cases, and log weak spots to your local Obsidian memory.

👉 **Get the TalentMe MCP Server**: [**https://github.com/airsota/talentme-mcp**](https://github.com/airsota/talentme-mcp)

---

## 🌐 Interactive Web Platform

Experience the full interactive visual suite online at [**talentme.airsota.com**](https://talentme.airsota.com):
* 🧭 **Dynamic Zoomable Mindmaps**: Explore 13 interactive module trees with 477 searchable nodes.
* 🃏 **Atomic Exam Cards**: Click any topic to reveal 5 core interview questions, common traps, and answers.
* 📊 **Modular Architecture Visualizer**: Toggle between SVG topology graphs and modern glassmorphic workflow blueprints.
* 🌗 **Dark & Light Mode**: Seamless bilingual instant toggle.

---

## 🤝 Contributing

We welcome community contributions, errata reports, and additional interview problem breakdowns!
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/awesome-addition`)
3. Commit your changes (`git commit -m 'Add new derivation for GRPO'`)
4. Push to the branch (`git push origin feature/awesome-addition`)
5. Open a Pull Request

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
Feel free to use these materials for personal study, interview preparation, and non-commercial educational purposes.

<div align="center">
  <sub>Built with ❤️ by the TalentMe AI Team</sub>
</div>
