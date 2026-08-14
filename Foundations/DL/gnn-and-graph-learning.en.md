---
title: "Graph Neural Networks (GNN) Taxonomy: Graph Laplacian, Message Passing (MPNN), GCN, GraphSAGE, GAT & Edge Feature Guide"
titleZh: "图神经网络 (GNN) 全景：邻接矩阵、图拉普拉斯矩阵、消息传递机制 (MPNN)、GCN、GraphSAGE、GAT 与边特征建模极客指南"
titleEn: "Graph Neural Networks (GNN) Taxonomy: Graph Laplacian, Message Passing (MPNN), GCN, GraphSAGE, GAT & Edge Feature Guide"
summaryZh: "100% 全量覆盖图论数学基础 (邻接矩阵 A、度矩阵 D、规范化图拉普拉斯矩阵 L_norm)、神经消息传递通用范式 (MPNN: Aggregate/Update/Readout)、GCN 谱域卷积与重归一化技巧 (Renormalization Trick)、GraphSAGE 归纳学习与邻域采样 (Inductive Sampling)、GAT 多头图注意力机制 (Multi-Head Attention)、边特征建模 (Edge-Conditioned Conv) 与 Node/Link/Graph 三类下游任务 Loss，以及 Pure Numpy GNN 算子引擎。配备丰富 SEO 长段说明文本。"
summaryEn: "100% exhaustive guide to Graph Neural Networks (GNN), covering graph mathematics (Adjacency matrix A, Degree matrix D, Normalized Laplacian L_norm), Neural Message Passing paradigm (MPNN: Aggregate/Update/Readout), GCN spectral graph convolution & Renormalization Trick, GraphSAGE inductive learning & neighborhood sampling, GAT Multi-Head Graph Attention, Edge feature modeling, downstream loss functions, and Pure Numpy GNN implementations with rich SEO explanatory text."
category: "foundations"
tags: ["deep-learning", "gnn", "gcn", "graphsage", "gat", "message-passing", "graph-laplacian", "seo-optimized"]
author: "TalentMe AI Team"
date: "2026-08-01"
interviewFollowups:
  - "Derive GCN symmetric normalized adjacency matrix and explain the Renormalization Trick."
  - "Compare Transductive GCN vs Inductive GraphSAGE neighborhood sampling for large graphs."
  - "Write down GAT attention coefficient alpha_vu and compare multi-head concatenation vs averaging."
  - "Explain the 3 Message Passing steps (Message, Update, Readout) and Over-smoothing causes."
  - "How are edge features incorporated in MPNN or Edge-Conditioned Conv for molecular graphs?"
---

# Graph Neural Networks (GNN) Taxonomy: Graph Laplacian, Message Passing (MPNN), GCN, GraphSAGE, GAT & Edge Feature Guide

> **Summary**: Representation learning on non-Euclidean graph-structured data is fundamental to modern recommender systems and molecular modeling. This 100% exhaustive guide covers graph matrices (Adjacency A, Degree D, Normalized Laplacian L_norm), Neural Message Passing (MPNN), GCN spectral convolution, GraphSAGE inductive sampling, GAT multi-head attention, and Pure Numpy GNN implementations with rich SEO explanatory text.

---

## 🧭 Knowledge Map & Architecture Graph

```mermaid
graph TD
    subgraph A["1. Graph Mathematics & Matrices"]
        A1["Adjacency Matrix A, Degree Matrix D"]
        A2["Normalized Graph Laplacian L_norm = I - D⁻¹/² A D⁻¹/²"]
        A3["Renormalization Trick: Ã = A + I_N"]
        A1 --> A2 --> A3
    end

    subgraph B["2. Message Passing Paradigm (MPNN)"]
        B1["Message Phase: m_v^(k) = AGGREGATE({h_u^(k-1) : u ∈ N(v)})"]
        B2["Update Phase: h_v^(k) = UPDATE(h_v^(k-1), m_v^(k))"]
        B3["Readout Phase: h_G = READOUT({h_v^(K) : v ∈ V})"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Core GNN Architectures"]
        C1["GCN: Symmetric normalization D̃⁻¹/² Ã D̃⁻¹/² H W"]
        C2["GraphSAGE: Fixed neighborhood sampling & Inductive learning"]
        C3["GAT: Dynamic Self-Attention alpha_vu & Multi-Head"]
        C4["MPNN / ECC: Edge features integration for molecular graphs"]
        C1 --> C2 --> C3 --> C4
    end

    A --> B --> C
```

> 💡 **Intuition**: A GNN layer is one "neighborhood meeting": each node sends a message, neighbors aggregate them, and the node updates itself from "old state + summary". $K$ layers = a node hears from $K$-hop neighbors — the graph version of receptive field. The four architectures differ only in the aggregator: GCN uses fixed degree-weighted averaging (needs the whole graph, transductive), GraphSAGE samples a fixed number of neighbors (constant compute, inductive — works on Pinterest-scale graphs), GAT learns dynamic attention weights (most expressive, slowest). Stacking too many layers (>4–6) makes every node converge to the same vector — over-smoothing — because normalized Laplacian powers squash everything into the principal eigenspace.
>
> 🎤 **Quick Answer**: "GCN's symmetric normalization $\tilde D^{-1/2}\tilde A \tilde D^{-1/2}$ weights an edge $1/\sqrt{\tilde D_{vv}\tilde D_{uu}}$ — a 100-degree node and a 2-degree node get weight ~0.071, avoiding degree explosion. GraphSAGE with $S_1=25, S_2=10$ keeps per-node cost constant ~250. GAT: hidden layers concatenate M heads, output layer averages. Cora: 2-layer GCN hits 81.5% node accuracy."

---

## 📚 Chapter 1: Pure Numpy GNN Engine

Plain-language reading (full implementations in the zh version): `gcn_layer_forward` is three lines — `A + np.eye(N)` adds self-loops, `1/sqrt(D_diag)` builds $\tilde D^{-1/2}$, sandwiching gives the symmetric normalization, then `ReLU(A_norm @ H @ W)` is one full GCN layer; `gat_layer_forward` splits attention scoring into a "sender score + receiver score" broadcast sum, masks non-neighbors to $-\infty$, and softmaxes.

```python
import numpy as np

class PureNumpyGNNEngine:
    @staticmethod
    def gcn_layer_forward(A: np.ndarray, H: np.ndarray, W: np.ndarray) -> np.ndarray:
        pass
    @staticmethod
    def graphsage_mean_forward(A: np.ndarray, H: np.ndarray, W: np.ndarray) -> np.ndarray:
        pass
    @staticmethod
    def gat_layer_forward(A: np.ndarray, H: np.ndarray, W: np.ndarray, a_vec: np.ndarray) -> np.ndarray:
        pass
```

> 💡 **Intuition**: All three share one skeleton — "neighbor matrix × features": GCN's `A_norm @ H` is static, GraphSAGE's `A_mean @ H` averages, GAT's `Alpha @ H_prime` is a dynamically learned attention matrix. Understand that one equation and the three architectures differ only in where the matrix comes from.
>
> 🎤 **Quick Answer**: "GNN forward = normalized adjacency × features × weights. The Renormalization Trick ($\tilde A = A + I$) both preserves the node's own features and bounds the spectral radius — that's why GCN works where plain $A$ diverges after a few layers."