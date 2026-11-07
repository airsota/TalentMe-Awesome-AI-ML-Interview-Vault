# 🖥️ AI Infrastructure · AI 基础设施 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **32 cards · 160 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. 4D Parallelism DP/TP/PP/SP · 4D 并行 DP/TP/PP/SP

> Four parallelism dimensions that shard parameters, gradients, activations and sequences across GPUs. ① DP (data parallel): every rank holds a full model copy, only data is sharded;

**Interview questions**

1. Derive DP's communication: the ring AllReduce per step is $2\frac{N-1}{N}S \approx 2S$ ($S$ = per-rank gradient), independent of N; explain why DP is still expensive for large models — S grows linearly with parameters (70B gradients ≈ 140GB).
2. TP communication profile: 2 AllReduces per layer, each ~$b \times s \times h$ and independent of parameter count; argue why TP must stay intra-node (NVLink 900GB/s, low latency) and what happens across nodes.
3. PP communication: why its volume is small (one activation + one gradient per microbatch between adjacent stages, ~$b \times s \times h$) and fits across nodes; what is its cost (bubble $\frac{p-1}{m+p-1}$, see its card).
4. SP: what is sharded (the sequence dimension), what it fixes (KV/activation memory linear in seq), and the communication cost and fit of Megatron-SP vs Ring-Attention.
5. 3D configuration problem: given total GPUs and model size, allocate TP intra-node → PP inter-node → DP outermost; recite GPT-3 175B's 1024 GPUs = DP 64 × TP 8 × PP 2 and justify each dimension.

<details>
<summary>中文题目</summary>

1. 手算 DP 通信量: 每步环形 AllReduce $2\frac{N-1}{N}S \approx 2S$($S$ = 单卡梯度), 与卡数 N 无关; 解释为什么大模型 DP 仍然贵——S 随参数量线性增长(70B 梯度 ≈ 140GB)。
2. TP 通信特征: 每层 2 次 AllReduce、每次消息量 ~$b \times s \times h$ 与参数量无关; 推导为什么 TP 只能在节点内(NVLink 900GB/s、低延迟), 跨节点会怎样。
3. PP 通信: 为什么 PP 通信量小(每 microbatch 相邻 stage 一次激活 + 一次梯度, ~$b \times s \times h$)且适合跨节点; 它的代价是什么(气泡率 $\frac{p-1}{m+p-1}$, 见对应卡片)。
4. SP 序列并行: 切什么(序列维度)、解决什么(KV/激活显存随 seq 线性)、两种实现 Megatron-SP 与 Ring-Attention 各自的通信代价与适用场景。
5. 3D 并行配置题: 给定 GPU 总数与模型规模, 按「TP 节点内 → PP 节点间 → DP 最外层」分配; 复述 GPT-3 175B 的 1024 卡 = DP 64 × TP 8 × PP 2 并解释每层选择的理由。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/distributed-parallel?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=distributed-parallel)

---

## 2. Activation Memory Estimation · 激活显存估算

> Activations are the intermediate outputs of each forward-pass layer; backprop reads them to compute gradients, so training must store or recompute them.

**Interview questions**

1. Derive the activation memory formula $\sum_l 2 \cdot s \cdot b \cdot h_l$: explain each factor (2 bytes/element FP16 × layer input $h_l$ elements × $s$ positions × $b$ samples), why it is independent of parameter count, and why it grows linearly with batch×seq.
2. Worked example: Llama-2 70B ($L=80$, $h=8192$), $s=4096$, $b=2$ → ≈134 MB/layer, ≈10.7 GB total; put it beside the static $16\Psi = 1120$ GB and state what each budget scales with (params vs batch×seq) — the complementary relationship.
3. Gradient checkpointing: keep a checkpoint every $k$ layers and recompute the forward pass in backward — activation memory drops from $O(L)$ to $O(\sqrt{L})$ at ~1/3 extra FLOPs; explain the memory-for-compute trade.
4. Which tensors dominate activation memory (the $s \times s$ attention score matrix, $4h$ MLP intermediates, dropout masks), and how selective recomputation reruns only the big ones to minimize overhead.
5. Distinguish activation memory (training) from KV cache (inference): activations are forward intermediates read back by backprop, growing with $L \times s \times b$; the KV cache stores generated K/V read sequentially once, formula $2 \cdot 2 \cdot L \cdot H \cdot d \cdot s \cdot b$; why the two budgets must not be conflated.

<details>
<summary>中文题目</summary>

1. 推导激活显存公式 $\sum_l 2 \cdot s \cdot b \cdot h_l$: 解释每项来源(FP16 每元素 2 字节 × 每层输入激活 $h_l$ 个元素 × $s$ 个位置 × $b$ 个样本),并说明它为什么与参数量无关、随 batch×seq 线性增长。
2. 数值例: Llama-2 70B($L=80$, $h=8192$), $s=4096$, $b=2$ → 每层 ≈ 134 MB、共 ≈ 10.7 GB;与静态 $16\Psi = 1120$ GB 并排讲清两本账各自随什么缩放(参数 vs batch×seq),即“互补”关系。
3. 梯度检查点(重计算)的原理与开销: 每 $k$ 层存一个检查点、反向重跑前向,激活显存从 $O(L)$ 降到 $O(\sqrt{L})$,FLOPs 增加约 1/3;说明为什么是“省显存换算力”。
4. 哪些张量最占激活显存(attention 的 $s \times s$ 分数矩阵、MLP 中间层 $4h$ 激活、dropout mask 等),选择性重计算如何只挑大头重算、把额外开销降到最低。
5. 激活显存(训练)与 KV Cache(推理)的区别: 前者是前向中间输出、反向必须回读、随 $L \times s \times b$ 增长;后者是自回归生成的 K/V、顺序只读一遍、公式 $2 \cdot 2 \cdot L \cdot H \cdot d \cdot s \cdot b$;为什么两本账不能混算。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/activation-memory?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=activation-memory)

---

## 3. Agent Runtime (cross-module) · Agent 运行时(跨模块)

> the agent runtime is the engineering backbone of the LLM agent loop — tool-execution sandbox (isolation/permissions/timeout), task queue and scheduling, and retry/idempotency control.

**Interview questions**

1. How do sandboxes isolate tool execution (permissions, timeouts) against malicious or runaway calls?
2. How do task queues and scheduling support high-concurrency agent requests (priority, concurrency caps, backpressure)?
3. Retry strategies and idempotency for failed tool calls (at-least-once vs exactly-once)?
4. How to cap resource usage (CPU/memory/network/filesystem) for sandboxed code execution?
5. How are timed-out or hung tool calls reclaimed to avoid task leaks?

<details>
<summary>中文题目</summary>

1. 工具执行沙箱如何实现隔离/权限/超时, 防止恶意或失控的工具调用?
2. 任务队列与调度如何支撑高并发 Agent 请求 (优先级、并发上限、背压)?
3. 工具调用失败的重试策略与幂等设计 (at-least-once vs exactly-once)?
4. 沙箱内代码执行如何限制资源 (CPU/内存/网络/文件系统)?
5. 超时或挂起的工具调用如何回收资源、避免任务泄漏?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/agent-runtime?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=agent-runtime)

---

## 4. Autoscaling & Cost · 弹性伸缩与成本优化

> Autoscaling = adding/removing GPU replicas based on live load — the valve between cost and latency. Three core signals: ① QPS — the most intuitive but the most lagging;

**Interview questions**

1. Choosing scaling signals: pros and cons of QPS, queue length / queuing latency and GPU utilization; why is queue backlog a better scale-up trigger than QPS (uneven request load)?
2. Why target GPU utilization at 70-90%: below 70% wastes idle capacity, above 90% queuing latency grows non-linearly; how does the target couple to the latency budget?
3. Autoscaling mechanics: K8s HPA custom metrics and cooldown to prevent oscillation; how do you mitigate scale-up cold-start latency (image pull / model load) with warm pools / preloaded models?
4. Spot instances: why are they 60-90% cheaper, how does reclamation work (about 2-minute warning), which workloads fit, and how do online services keep SLA with spot + on-demand hybrid pools?
5. The cost-optimization playbook: how spot, reserved instances, utilization monitoring and off-peak batch processing stack; where does a full cost-optimization pass start?

<details>
<summary>中文题目</summary>

1. 伸缩指标选型: QPS、队列长度/排队延迟、GPU 利用率各自的优劣;为什么队列积压比 QPS 更适合触发扩容(请求负载不均)?
2. GPU 利用率为什么定在 70-90%: 低于 70% 空转浪费、高于 90% 排队延迟非线性飙升;目标值与延迟预算如何挂钩?
3. 扩缩容实现: K8s HPA 自定义指标与 cooldown 防抖动;扩容的冷启动延迟(镜像拉取/模型加载)如何缓解(预热池/预加载模型)?
4. Spot 实例: 为什么能便宜 60-90%,回收机制(约 2 分钟预警)是什么;哪些负载适合 spot,在线服务如何用 spot + 按需混合池保证 SLA?
5. 成本优化组合拳: spot、预留实例、利用率监控、off-peak 批处理如何叠加;一次完整的成本优化流程从哪一步开始?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/autoscaling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=autoscaling)

---

## 5. Checkpointing & Recovery · 检查点与故障恢复

> A checkpoint is a complete snapshot of training state so that thousand-GPU runs can resume from the latest step after failures (node crash / OOM / power loss) instead of starting over.

**Interview questions**

1. List everything a checkpoint must save (weights, Adam $m_t$/$v_t$, step/epoch and LR schedule, RNG states, DataLoader cursors) and explain why saving only weights makes the resumed loss unstable (lost optimizer state and LR progress).
2. What sharded checkpointing fixes: a single 140 GB FP16 70B file takes minutes to persist and is fragile; how HF safetensors sharding (index.json + parallel writes + mmap zero-copy reads) solves it and why it is safer than pickle (.bin).
3. Async checkpointing: why synchronous saves must stop-the-world (weights keep changing); how async saves obtain a consistent snapshot (GPU→CPU copy + copy-on-write/versioning + background persistence) and the trade-offs.
4. How to save/resume under ZeRO-3/FSDP sharding: each rank persists its own $16\Psi/N_d$ shard and reads it back on resume (or gathers a consolidated file); why RNG and shuffle order must be restored as well.
5. Training stability: how to detect and handle loss/gradient spikes (roll back to a last-good checkpoint, lower LR, skip the bad batch), how to trade off save frequency (IO cost vs failure loss), with a per-epoch candidate-checkpoint strategy.

<details>
<summary>中文题目</summary>

1. 完整列出 checkpoint 必须保存的状态(权重、Adam $m_t$/$v_t$、step/epoch 与 LR 调度、RNG、DataLoader 游标),并解释为什么只存权重会导致恢复后 loss 波动(优化器状态与学习率进度丢失)。
2. 碎片保存解决什么: 70B FP16 单文件 140 GB 分钟级落盘且易损坏;HF safetensors sharded 如何用 index.json + 多分片并行写 + mmap 零拷贝读解决,为什么比 pickle(.bin)安全。
3. 异步保存原理: 为什么同步保存必须 stop-the-world(权重持续变化、无法直接后台写);异步保存如何拿一致快照(GPU→CPU 拷贝 + copy-on-write/版本戳 + 后台落盘)以及权衡。
4. ZeRO-3/FSDP 分片训练下如何保存与恢复: 每 rank 只存自己的分片 $16\Psi/N_d$,恢复时逐 rank 读回(或先 gather 拼全量导出);续训时 RNG 与 shuffle 为何必须一并恢复。
5. 训练稳定性: loss/梯度尖刺如何检测与应对(回滚 last-good 检查点、降 LR、跳过异常 batch),保存频率如何权衡(IO 开销 vs 故障损失),结合每 epoch 候选检查点给出策略。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/checkpoint-resume?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=checkpoint-resume)

---

## 6. Cluster Scheduling Ray/K8s · 集群调度 Ray/K8s

> Cluster scheduling answers 'which task goes to which GPU and how to avoid wasting resources'. The K8s default scheduler (kube-scheduler) is a two-phase pipeline: ① Filtering: drop nodes that violate hard constraints — resource requests (CPU/memory/extended res…

**Interview questions**

1. Describe the K8s scheduler's two phases: the hard constraints checked in filtering (resources, NodeSelector, affinity, taints/tolerations, ports/volumes), the scoring policies (least-requested vs most-requested) and their trade-offs, and why filtering must come first.
2. Why GPU is an extended resource allocated in whole cards (memory and SMs cannot be safely partitioned), how the device plugin advertises $nvidia.com/gpu$; can a 30 GB job fit a card with only 20 GB free, and why.
3. What is MIG: hardware slicing of A100/H100 (up to 7 instances, e.g. $1g.5gb$) with dedicated SM and memory partitions per instance; the key difference from vGPU time-slicing (isolation, no jitter), and when to choose it.
4. Sketch the Ray architecture: Driver → Head (GCS global control plane + Raylet) → per-node Raylet (local scheduler + Object Store) → Workers; why the object store gives zero-copy sharing between same-node tasks while only cross-node traffic hits the network.
5. Autoscaling: the Ray Autoscaler adds/removes nodes by pending task counts, K8s via HPA/custom metrics; how in-flight work is handled gracefully on scale-down and spot preemption (drain, replicas, checkpoint-resume).

<details>
<summary>中文题目</summary>

1. 描述 K8s 调度两阶段: 过滤检查哪些硬约束(资源、NodeSelector、亲和性、污点容忍、端口/卷),打分有哪些策略(least-requested / most-requested)及其取舍,为什么必须先过滤后打分。
2. GPU 为什么是扩展资源、为什么只能整卡调度(显存与 SM 不可安全分片),device plugin 如何上报 $nvidia.com/gpu$;只剩 20 GB 显存的卡能否容纳需要 30 GB 的任务,为什么。
3. MIG 是什么: A100/H100 硬件切片(最多 7 个实例,如 $1g.5gb$),每个实例独享 SM 与显存分区;与 vGPU 时间片共享的本质区别(隔离性与性能抖动),什么场景选 MIG。
4. 画出 Ray 架构: Driver → Head(GCS 全局控制面 + Raylet)→ 每节点 Raylet(本地调度 + Object Store)→ Worker;对象存储为什么让同节点任务零拷贝共享数据、只有跨节点才走网络。
5. 弹性扩缩: Ray Autoscaler 按待处理任务数增删节点、K8s HPA/自定义指标伸缩;缩容与 spot 回收时如何优雅处理在跑任务(drain、多副本、检查点续训)。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/cluster-scheduling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cluster-scheduling)

---

## 7. Collective Comm & NVLink Topology · 集合通信与 NVLink 拓扑

> Every synchronization in multi-GPU training reduces to collective primitives. Let each rank hold a data shard of $S$ bytes and there be N ranks: ① AllGather (each rank broadcasts its shard to the other N−1 ranks;

**Interview questions**

1. Derive AllGather's volume: shard $S$ per rank, N ranks, each ends with $NS$ and must receive $(N-1)S$ new bytes — explain why it grows linearly with N while Ring-AllReduce does not.
2. Decompose AllReduce into ReduceScatter + AllGather: what is each phase's volume ($\frac{N-1}{N}S$ and $(N-1)S$) and step count (N−1); verify the total $2\frac{N-1}{N}S$ by addition.
3. Why MoE expert parallelism is AllToAll, not AllGather: tokens route by expert ID, each rank sends to and receives from all ranks; compare volume $\frac{N-1}{N}S$ under uniform load with AllReduce.
4. Recite NVLink/IB numbers: NVLink 4.0 = 900GB/s, NVLink 5.0 = 1.8TB/s, IB 800Gbps = 100GB/s, 400Gbps = 50GB/s; compute how many × NVLink 5.0 is over IB 800Gbps (≈18×) and why TP must use NVLink while DP can use IB.
5. Describe rail-optimized topology: the i-th GPU of every node plugs into one shared leaf switch forming a “rail”; what problem it solves (same-index communication, no cross-rail hotspots) and its relation to NVSwitch and leaf-spine.

<details>
<summary>中文题目</summary>

1. 手算 AllGather 通信量: 每卡分片 $S$、N 卡, 每卡最终持有 $NS$, 需新接收 $(N-1)S$ 字节——推导它为什么随 N 线性增长而 Ring-AllReduce 不随 N 增长。
2. 把 AllReduce 拆成 ReduceScatter + AllGather 两阶段: 各阶段数据量($\frac{N-1}{N}S$ 与 $(N-1)S$)与步数(N−1)各是多少, 相加验证总量 $2\frac{N-1}{N}S$。
3. MoE 专家并行为什么是 AllToAll 而不是 AllGather: token 按专家 ID 路由、每卡发给所有卡也接收所有卡; 均匀负载下通信量 $\frac{N-1}{N}S$ 与 AllReduce 的对比。
4. 背 NVLink/IB 数字: NVLink 4.0 = 900GB/s、NVLink 5.0 = 1.8TB/s、IB 800Gbps = 100GB/s、400Gbps = 50GB/s; 算 NVLink 5.0 是 IB 800Gbps 的多少倍(≈18×), 并说明为什么 TP 必须走 NVLink 而 DP 可以走 IB。
5. 描述轨优化(rail-optimized)拓扑: 每节点第 i 张 GPU 接同一台叶交换机构成“轨”; 它解决了什么问题(同号卡之间通信、避免跨轨横跳热点), 与 NVSwitch、叶脊拓扑的关系。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/collective-communication-nvlink?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=collective-communication-nvlink)

---

## 8. Context & Memory Services · 上下文缓存与记忆服务

> context and memory services reuse prefix KV caches across requests (prefix KV caching service) and provide tiered memory — short-term working memory plus long-term storage/retrieval.

**Interview questions**

1. How does prefix KV caching reuse work across requests, and how does hit rate affect TTFT and cost?
2. How are short-term and long-term memory tiered (working memory vs external vector store)?
3. How do retrieval services (RAG) integrate with memory services, trading off relevance vs latency?
4. How to control long-session cost (window trimming, summarization, cached prefixes)?
5. How are prefix caches invalidated and rebuilt when system prompts or history change?

<details>
<summary>中文题目</summary>

1. 前缀 KV 缓存 (prompt caching) 如何跨请求复用, 命中率如何影响 TTFT 与成本?
2. 短期记忆与长期记忆如何分层存储 (工作记忆 vs 外部向量库)?
3. 检索服务 (RAG) 如何与记忆服务集成, 相关度与延迟如何权衡?
4. 长会话成本如何控制 (窗口裁剪、摘要压缩、缓存前缀)?
5. 系统提示或历史更新后, 前缀缓存如何失效与重建?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/context-memory-service?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=context-memory-service)

---

## 9. Continuous Batching · 连续批处理

> Continuous Batching (iteration-level scheduling, introduced by Orca) does not wait for an entire batch to finish before serving the next one;

**Interview questions**

1. Static vs continuous batching: draw a timeline with two requests (10 tokens vs 40 tokens) showing static batching's idle time vs continuous batching's refill, and why throughput rises 2-3×.
2. Iteration-level scheduling: what the engine does at the end of each decode iteration (evict finished, insert prefilled newcomers, dynamic membership); why a request must prefill before joining decode.
3. Preemption and swap: who gets preempted under memory pressure (new prefill vs running decode), where evicted KV goes (CPU swap / discard-restart), and each option's latency cost.
4. Why workloads with many short requests benefit most: fast completion, fast slot turnover; relate to the padding/chunked-attention cost of unequal-length sequences in a batch.
5. How continuous batching composes with PagedAttention and PD disaggregation: paged KV makes dynamic membership fragmentation-free; how scheduling works in prefill nodes vs decode nodes.

<details>
<summary>中文题目</summary>

1. 静态批 vs 连续批的差异: 用两个具体请求(一个 10 token、一个 40 token)画时间线,说明静态批的空转与连续批的填充逻辑,以及吞吐为何可提升 2-3×。
2. 迭代级调度机制: 每个解码迭代结束时引擎做什么(完成移出/新请求 prefill 后插入/批成员动态变化);为什么新请求必须先 prefill 才能 decode。
3. 抢占 (preemption) 与 swap: 显存不足时抢占谁(新 prefill vs 旧 decode)、被抢占请求的 KV 放哪里(CPU swap / 丢弃重启),各自延迟代价是什么。
4. 为什么短请求占比高的负载收益最大: 短请求完成快、slot 周转快;结合批内序列长度参差时的 padding/分块注意力代价说明取舍。
5. 连续批如何与 PagedAttention、PD 分离配合: 分页 KV 使动态进出批无碎片;prefill/decode 分离后连续批分别在两阶段如何调度。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/continuous-batching?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=continuous-batching)

---

## 10. Data Loading & IO · 训练数据加载与 IO

> Training data loading and IO optimization. WebDataset packs samples sequentially into tar shards read as streams, replacing random reads of millions of small files with sequential IO;

**Interview questions**

1. Why do many small random reads kill data loading? How does WebDataset's sharded tar fix it?
2. When to use TFRecord vs Parquet vs Arrow?
3. How to tell whether the bottleneck is data loading or compute? Which tools pinpoint it?
4. How do prefetch and multi-worker loading overlap IO with compute? Ideal total time?
5. How are data sharded per rank and globally shuffled across epochs in distributed training?

<details>
<summary>中文题目</summary>

1. 为什么海量小文件随机读会打爆数据加载? WebDataset 的 shard 机制如何解决?
2. TFRecord / Parquet / Arrow 各自适合什么场景? 如何选择?
3. 如何判断训练瓶颈在数据加载而非算力? 用什么工具定位?
4. 预取与多进程 worker 如何实现 IO 与计算重叠? 理想重叠下的总耗时是多少?
5. 分布式训练中数据如何按 rank 分片、跨 epoch 全局打乱并保证可复现?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/data-loading?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=data-loading)

---

## 11. Data Pipelines & Streaming · 数据管线与流处理

> Data pipelines and streaming. Kafka acts as the message bus (ordered per partition, append-only writes + page cache + zero-copy for high throughput;

**Interview questions**

1. Why is Kafka high-throughput? Partitions, sequential writes and zero-copy; how are at-least-once and exactly-once achieved?
2. How does Flink achieve exactly-once? Checkpoint barriers and two-phase commit?
3. Lambda vs Kappa: tradeoffs and why Kappa is increasingly preferred?
4. Responsibilities of ODS/DWD/DWS and the data flow? Where do redaction and metric unification happen?
5. How to unify online and offline features for train-serve consistency?

<details>
<summary>中文题目</summary>

1. Kafka 为什么吞吐高? 分区、顺序写与零拷贝的机制; at-least-once 与 exactly-once 怎么实现?
2. Flink 如何保证精确一次? checkpoint/barrier 与两阶段提交的原理是什么?
3. Lambda 与 Kappa 架构的取舍? 为什么 Kappa 越来越流行?
4. ODS/DWD/DWS 各层职责与数据流转? 脱敏与指标口径统一在哪一层完成?
5. 实时特征与离线特征如何统一, 保证训练-服务特征一致?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/data-pipeline-streaming?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=data-pipeline-streaming)

---

## 12. Dataset Management & Lineage · 数据集管理与血缘

> Dataset management systematically governs the versioning, lineage, storage, compliance and lifecycle of training data. The core tool is DVC (Data Version Control): data files stay out of Git while the version store keeps only metadata (file hashes, dependencie…

**Interview questions**

1. How does DVC version large datasets? Why keep data files out of Git, and how are metadata and data separated?
2. Why does data lineage matter? How do you record and trace the data-to-feature-to-model chain?
3. When to use exact-hash vs MinHash/LSH dedup? What is the hash-collision probability formula?
4. What does training-data compliance cover? How to implement PII redaction, licensing and cross-border rules?
5. Data lifecycle: how to design retention, archive and deletion policies, and how do they tie to model versions?

<details>
<summary>中文题目</summary>

1. DVC 如何实现大数据集版本管理? 为什么数据文件不入 Git, 元数据与数据如何分离?
2. 数据血缘 (Lineage) 的价值是什么? 如何记录与追溯「数据 → 特征 → 模型」链路?
3. 训练数据去重的两类方法(精确哈希 vs MinHash/LSH 近似)各适合什么场景? 哈希碰撞概率公式是什么?
4. 训练数据合规包含哪些要点? PII 脱敏、版权许可、数据出境如何落地?
5. 数据生命周期管理: 保留、归档与删除策略如何设计? 数据版本与模型版本如何关联?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/dataset-management?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dataset-management)

---

## 13. FlashAttention Kernel & Online Softmax · FlashAttention 与 Online Softmax

> FlashAttention is a GPU-memory-optimized attention kernel: by tiling $Q/K/V$ into SRAM and merging partial softmaxes via Online Softmax, HBM traffic drops from $O(N^2)$ to $O(N^2 / M)$ (≈ $O(N)$ for large $N$; $M$ = SRAM block capacity).

**Interview questions**

1. Derive standard attention's HBM bottleneck: why the $N \times N$ round-trip between $S = \frac{QK^\top}{\sqrt{d}}$ and softmax costs $O(N^2)$ traffic; why bandwidth dominates at long sequences (e.g. 128K)?
2. Write the Online Softmax updates: $m_{new} = \max(m, x_i)$, $l_{new} = l \cdot e^{m - m_{new}} + e^{x_i - m_{new}}$; why do you need a running max with rescaling instead of directly accumulating $e^{x_i}$?
3. Tiling details: what limits block size (e.g. ~228KB SRAM per SM on H100)? How does HBM traffic drop from $O(N^2)$ to $O(N^2 / M)$; how is the causal mask handled inside the kernel?
4. FA-1/2/3 evolution: why backward recomputation cuts memory from $O(N^2)$ to $O(N)$; FA-2's head parallelism and upcast optimization; what FA-3's warp specialization/TMA async and FP8 solve?
5. Relation between FlashAttention and KV cache/PagedAttention: why kernel-level optimization composes with engine-level optimization; why the decode bandwidth bottleneck (reading weights per token) cannot be fixed by FA alone?

<details>
<summary>中文题目</summary>

1. 推导标准注意力的 HBM 瓶颈: 为什么 $S = \frac{QK^\top}{\sqrt{d}}$ 与 softmax 之间的 $N \times N$ 矩阵往返是 $O(N^2)$ 访存;长序列(如 128K)下为什么带宽主导?
2. 写出 Online Softmax 增量公式: $m_{new} = \max(m, x_i)$、$l_{new} = l \cdot e^{m - m_{new}} + e^{x_i - m_{new}}$;为什么需要 running max + 重缩放而不能直接累加 $e^{x_i}$?
3. Tiling 细节: 块大小受什么限制(每 SM SRAM 如 H100 约 228KB)?HBM 访问如何从 $O(N^2)$ 降到 $O(N^2 / M)$;causal mask 如何在 kernel 内处理?
4. FA-1/2/3 演进: 反传重算为何把显存从 $O(N^2)$ 降到 $O(N)$;FA-2 的多头并行与 upcast 优化;FA-3 的 warp specialization/TMA 异步与 FP8 各解决什么?
5. FlashAttention 与 KV cache/PagedAttention 的关系: kernel 层优化为何不与推理引擎层优化冲突;为什么 decode 阶段带宽瓶颈(逐 token 读权重)不能靠 FA 解决?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/flashattention-kernel?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=flashattention-kernel)

---

## 14. GPU Hardware & HBM · GPU 硬件与 HBM

> The hardware foundation of GPU training/inference is a three-axis trade-off: compute, memory bandwidth, and memory capacity.

**Interview questions**

1. Recite H100 specs: 80GB HBM3, 3.35TB/s bandwidth, 989 TFLOPS FP16, 50MB L2; compute the balance arithmetic intensity $\approx 295\,\text{FLOP/byte}$ and classify the GPU as compute- or bandwidth-bound.
2. Hand-derive the memory wall: 70B FP16 weights = 140GB, one pass costs $\frac{140\,\text{GB}}{3.35\,\text{TB/s}} \approx 42\,\text{ms}$; one token's forward is $2 \times 70\text{B}$ FLOP = $\approx 0.14\,\text{ms}$ at 989 TFLOPS; ~300:1 ratio, explaining why training is bandwidth-bound.
3. Sketch the memory hierarchy: registers → SRAM/shared memory → L2 → HBM → host memory with typical capacity/bandwidth orders (H100: 228KB/SM, 50MB, 80GB); why FlashAttention keeps intermediates in SRAM instead of writing back to HBM.
4. Use arithmetic intensity $I=\text{FLOP/byte}$ to classify ops: state H100's balance point $\approx 295$; give bandwidth-bound examples (LayerNorm/activations/GELU) vs compute-bound (large matmul) and explain the reasoning.
5. Argue from bandwidth/capacity: why 70B cannot fit one card (140GB weights > 80GB HBM), requiring weight sharding (TP/ZeRO) or quantization; why byte-reduction (quantization, recompute) beats raw FLOPs when weights are re-read wholesale.

<details>
<summary>中文题目</summary>

1. 报 H100 关键数字: 80GB HBM3、3.35TB/s 带宽、989 TFLOPS FP16、L2 50MB; 口算平衡算术强度 $\approx 295\,\text{FLOP/byte}$ 并判断它是算力受限还是带宽受限。
2. 手算内存墙: 70B FP16 权重 140GB, 搬运一遍 $\frac{140\,\text{GB}}{3.35\,\text{TB/s}} \approx 42\,\text{ms}$; 单 token 前向 $2 \times 70\text{B}$ FLOP 在 989 TFLOPS 下 $\approx 0.14\,\text{ms}$; 比值约 300:1, 由此说明训练为何带宽受限。
3. 画出显存层级: 寄存器 → SRAM/共享内存 → L2 → HBM → 主机内存, 每级典型容量与带宽量级(H100: 228KB/SM、50MB、80GB); FlashAttention 为什么要把中间结果留在 SRAM 而不写回 HBM。
4. 用算术强度 $I=\text{FLOP/byte}$ 判断算子类型: 给出 H100 平衡点 $\approx 295$; 举出带宽受限(LayerNorm/激活/GELU)与算力受限(大 matmul)的例子并解释判断过程。
5. 从带宽/容量角度论证: 为什么 70B 单卡放不下(权重 140GB > 80GB), 需要权重切分(TP/ZeRO)或量化; 为什么权重反复全量读出的场景优先优化字节数(量化、重计算)而非算力。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/gpu-hardware?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gpu-hardware)

---

## 15. High-Concurrency Serving · 高并发与负载均衡

> High-concurrency LLM serving = capacity planning and traffic governance that sustain a target QPS on finite GPU compute while honoring latency budgets (TTFT, TBT/TPOT, end-to-end P50/P95).

**Interview questions**

1. Hand-compute capacity: GPU at 2000 tokens/s, average output 500 tokens, TBT budget 20 ms → max concurrency $2000/50 = 40$; replicas needed for a target QPS; why latency degrades past ~80% utilization (M/M/c).
2. Token-bucket rate limiting: how to set capacity $C$ (burst) and refill rate $r$ (average QPS); handling bursts vs steady traffic; reject vs queue on overflow and the reasoning.
3. Queuing and backpressure: why the queue must be bounded; options when full (503 reject, backoff retry, degrade); how backpressure prevents retry storms and cascades, and the role of timeout circuit breaking.
4. Multi-replica routing: round-robin vs least-connections vs consistent hashing; why prefix-cache affinity calls for consistent hashing (pinning a session to a replica), and how failed replicas are drained.
5. Latency budgets and graceful degradation: how to split TTFT/TBT/P95 budgets; what priority queues, output truncation and small-model fallback sacrifice vs protect under overload.

<details>
<summary>中文题目</summary>

1. 容量规划手算: 单卡 2000 tokens/s、平均输出 500 token、TBT 预算 20ms,算出单副本最大并发($2000/50 = 40$)与给定目标 QPS 所需副本数;解释为什么利用率超过 ~80% 后延迟恶化。
2. 令牌桶限流: 桶容量 $C$(突发)与补充速率 $r$(平均 QPS)如何设定;突发请求与平滑流量的处理差异;超过时拒绝还是排队,依据是什么。
3. 排队与背压: 队列为什么必须有界;队列满时的选择(503 拒绝、退避重试、降级);背压如何防止重试风暴与雪崩,超时熔断的作用。
4. 多副本路由策略: round-robin vs least-connections vs 一致性哈希;为什么前缀缓存亲和需要一致性哈希(会话固定副本),副本故障时如何摘除。
5. 延迟预算体系与优雅降级: TTFT/TBT/P95 如何分解预算;优先级队列、输出截断、降级到小模型等过载策略分别牺牲什么保住什么。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/high-concurrency?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=high-concurrency)

---

## 16. Inference Engines vLLM/SGLang/TensorRT-LLM · 推理引擎 vLLM/SGLang/TensorRT-LLM

> An inference engine is the system-software layer that packages model weights, KV-cache scheduling, batching and kernel optimization into an efficient serving stack, maximizing GPU utilization, throughput and latency (TTFT/TBT) targets.

**Interview questions**

1. Compare vLLM vs SGLang vs TensorRT-LLM core mechanisms: PagedAttention / RadixAttention / graph optimization + quantization; which workload suits each (TTFT-sensitive? throughput ceiling? flexibility?).
2. Why Continuous Batching is standard in every engine: vs static batching, iteration-level scheduling prevents fast requests being blocked by slow ones, 2-3× throughput; walk through per-step completion/insertion/preemption.
3. Quantization backend choice: FP8 (E4M3, H100-native, ~2× FP16 throughput) vs AWQ (activation-aware, INT4 near-FP16 quality) vs INT4/INT8 (half/quarter memory) precision/speed/memory trade-offs.
4. Why prefill/decode disaggregation boosts throughput and stability: prefill is compute-bound, decode bandwidth-bound, mixed they interfere; separated nodes each run optimal batch sizes — how TTFT and TBT benefit respectively.
5. Evaluate engines by metrics: define and measure TTFT / TBT / tokens/s throughput / P95; given per-GPU peak throughput and average output length, estimate sustainable concurrency and QPS.

<details>
<summary>中文题目</summary>

1. vLLM vs SGLang vs TensorRT-LLM 核心机制对比: PagedAttention / RadixAttention / 图优化+量化;各自最适合什么负载(TTFT 敏感?吞吐极限?灵活性?)。
2. 为什么 Continuous Batching 是引擎标配: 对比静态批,迭代级调度下快请求不被慢请求阻塞,吞吐可提升 2-3×;解释完成/插入/抢占的每步逻辑。
3. 量化后端选型: FP8(E4M3, H100 原生,吞吐约 2×FP16)vs AWQ(激活感知,INT4 精度接近 FP16)vs INT4/INT8(显存减半/1/4)的精度/速度/显存取舍。
4. PD 分离为什么提升吞吐与稳定性: prefill 算力密集、decode 带宽密集,混跑互相干扰;分离后各自节点最优批大小,解释 TTFT 与 TBT 如何分别受益。
5. 用指标评估引擎: TTFT / TBT / 吞吐 tokens/s / P95 的定义与测量;给定单卡峰值吞吐与平均输出长度,估算可支撑的并发与 QPS。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/inference-engine?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=inference-engine)

---

## 17. Inference Quantization (cross-module) · 推理量化(跨模块)

> Inference quantization = casting model weights (and optionally activations and KV cache) from FP16 to INT8/INT4/FP8 to trade memory for serving throughput (quantization theory and error analysis live in the LLM module;

**Interview questions**

1. Difference between W8A8 and W4A16? Why is activation quantization harder than weight quantization (outliers) and how does SmoothQuant migrate outliers into the weights?
2. How AWQ works: why do activation-aware scales on "important channels" preserve accuracy? Why is calibration cheaper than GPTQ (no backprop)?
3. FP8 serving: E4M3 layout and native H100 support; vs INT8 it has wider dynamic range and needs less calibration; when would you still pick INT4?
4. Serving math: 7B FP16 ≈ 14GB → INT8 ≈ 7GB → INT4 ≈ 3.5GB; how do batch size and throughput scale in the same memory? How do group sizes 128/32 trade precision and overhead?
5. Memory-precision-throughput trade-off: when to choose post-training quantization (PTQ) vs quantization-aware training (QAT)? How do you verify quantization loss stays within quality gates before rollout?

<details>
<summary>中文题目</summary>

1. W8A8 与 W4A16 的区别?为什么激活量化比权重量化难(离群值),SmoothQuant 如何把离群从激活迁移到权重?
2. AWQ 原理: 为什么按激活幅度感知给“重要通道”更大的 scale 能保精度?与 GPTQ 相比为什么校准更便宜(无需反向传播)?
3. FP8 服务侧: E4M3 布局、H100 原生支持;与 INT8 相比动态范围更大、校准需求更小;什么场景仍需要 INT4?
4. 服务落地账: 7B FP16 ≈ 14GB → INT8 ≈ 7GB → INT4 ≈ 3.5GB;同样显存下批大小与吞吐如何变化?group size 128/32 如何影响精度与开销?
5. 显存-精度-吞吐权衡: 训练后量化(PTQ)与量化感知训练(QAT)各自何时选?上线前如何验证量化精度损失在质量门禁内?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/quantization-inference?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=quantization-inference)

---

## 18. KV Cache & PagedAttention · KV Cache 与 PagedAttention

> KV Cache is the inference memory structure that caches the Key/Value vectors of already-generated tokens during autoregressive decoding, avoiding recomputing attention over the full history at every step.

**Interview questions**

1. Hand-compute KV cache memory: write $\text{KV Bytes} = 2\cdot2\cdot n_{layers}\cdot n_{kv\_heads}\cdot d_{head}\cdot s\cdot b$ and explain what each 2 means; for 80 layers, 8 KV heads, $d_{head}=128$, 16K context, batch 16, compute per-sequence and total GB.
2. Why PagedAttention removes memory fragmentation: 16-token blocks plus a block table; how internal and external fragmentation arise and disappear; utilization improvement from ~60-80% to ~100%.
3. How GQA/MLA compress KV memory: which term in the formula shrinks when MHA → GQA shares KV heads; LLaMA-3 70B (64→8) gives an 8× reduction; MLA's low-rank projection idea.
4. KV write patterns in prefill vs decode: prefill writes all s KV entries at once, decode appends one per step; why the memory peak sits at the end of prefill.
5. Precision trade-offs of KV quantization: why quantizing KV (e.g. INT8) is more sensitive than weight quantization; how it composes with FlashAttention and streaming/sliding-window KV to cap memory.

<details>
<summary>中文题目</summary>

1. 手算 KV cache 显存: 写出公式 $\text{KV Bytes} = 2\cdot2\cdot n_{layers}\cdot n_{kv\_heads}\cdot d_{head}\cdot s\cdot b$ 并解释两个 2 的含义;给定 80 层、8 个 KV heads、$d_{head}=128$、16K 上下文、batch 16,算每序列/整批多少 GB。
2. PagedAttention 为什么能消除显存碎片: 16 token/块的页表机制;内部碎片与外部碎片分别怎么产生、分页后如何消失;显存利用率从约多少提升到多少。
3. GQA/MLA 如何压缩 KV 显存: MHA → GQA 共享 KV heads 后公式里哪一项变小;LLaMA-3 70B (64→8) 对应 8 倍降幅;MLA 更进一步(低秩投影压缩)的原理。
4. Prefill 与 Decode 阶段 KV 缓存的写入与复用差异: prefill 一次性写入全部 s 个 KV、decode 每步追加一个;为什么长 prompt 的显存峰值出现在 prefill 末尾。
5. KV 缓存与量化/压缩的精度权衡: 为什么 KV 量化 (如 INT8) 比权重量化更敏感;与 FlashAttention、StreamingLLM 滑动窗口如何组合控制显存。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/kv-cache-paged?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=kv-cache-paged)

---

## 19. KV Cache Optimization · KV 缓存优化

> KV Cache optimization compresses the K/V tensors cached per layer during autoregressive inference to break the memory wall of long contexts and large batches.

**Interview questions**

1. Write the KV cache memory formula $2 \times 2 \times L \times H \times d \times s \times b$ and explain each factor; roughly how many GB does a 70B (e.g. 80 layers, 64 heads, head dim 128) use at 8K context with batch 1?
2. GQA vs MHA/MQA: why sharing KV heads (e.g. 64 → 8) barely hurts accuracy (attention head redundancy) and how much memory it saves?
3. MLA's low-rank compression: how does $c_t^{KV} = W^{DKV} h_t$ compress per-layer KV into a latent; why can the $W^Q$/$W^U$ projections be fused at inference; the 86GB → 5.1GB numbers?
4. KV quantization (INT8/FP8): why are K/V easier to quantize than activations (stable value ranges); the cost of halving memory and how it composes with GQA/MLA?
5. Why must StreamingLLM keep the attention sink (softmax normalization stability, first 4 tokens); how does H2O keep heavy-hitter tokens by cumulative attention score, and how does it differ from sliding windows?

<details>
<summary>中文题目</summary>

1. 写出 KV Cache 显存公式 $2 \times 2 \times L \times H \times d \times s \times b$ 并解释每个因子的含义;70B(如 80 层、64 头、128 头维)在 8K 上下文、批 1 时约为多少 GB?
2. GQA 与 MHA/MQA 的对比: 为什么共享 KV 头(如 64 → 8)几乎不掉精度(注意力头冗余),显存降多少?
3. MLA 的低秩压缩: $c_t^{KV} = W^{DKV} h_t$ 如何把逐层 KV 压成潜在向量;推理时为什么能吸收进 $W^Q$/$W^U$ 融合;对照数字 86GB → 5.1GB?
4. KV 量化(INT8/FP8): 为什么 K/V 比激活更适合量化(数值范围稳定);显存减半的代价与 GQA/MLA 如何叠加?
5. StreamingLLM 的 attention sink 为什么必须保留(softmax 归一化稳定性,前 4 个 token);H2O 如何按累计注意力分数保留 heavy-hitter token,与滑动窗口的区别?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/kv-cache-optimization?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=kv-cache-optimization)

---

## 20. Megatron 3D & Pipeline Bubble · Megatron TP/PP 与气泡率

> Megatron-LM's 3D parallelism (TP × PP × DP) implementation. TP column/row parallel with dual AllReduce: inside each transformer layer, the attention QKV projection is column-parallel (output split along hidden), the output projection row-parallel (input split…

**Interview questions**

1. Hand-derive the bubble: p stages, m microbatches → $\text{bubble}=\frac{p-1}{m+p-1}$; give numeric examples (p=4,m=4 → 3/7≈43%; p=8,m=32 → 7/39≈18%) and how to reduce it (larger m, interleaved chunks).
2. Describe TP column/row parallel with dual AllReduce: why column-parallel outputs need AllReduce summation and why row-parallel backward does too; how many per layer per iteration (2 forward + 2 backward = 4).
3. Compare greedy vs 1F1B scheduling: why 1F1B has lower peak memory (in-flight microbatches ≤ p) and earlier backprop start; why the bubble is not eliminated (inter-stage pipeline stalls).
4. Derive Ring-Attention: why per-rank KV memory is O(1) after splitting the sequence into N chunks; per-layer communication ≈ 2× the full KV; bandwidth/latency profile (constant bandwidth, N−1 serial hops) and its relation to FlashAttention.
5. 3D integration: explain the DP/TP/PP division by communication level (TP intra-node frequent small messages, PP inter-node infrequent large messages, DP gradient AllReduce), and why 3D parallelism grows parallelism while cutting per-GPU memory and communication cost.

<details>
<summary>中文题目</summary>

1. 手算气泡率: p 个 stage、m 个 microbatch → $\text{bubble}=\frac{p-1}{m+p-1}$; 给出数值例(p=4,m=4 → 3/7≈43%; p=8,m=32 → 7/39≈18%)并说明如何降低气泡(增大 m、interleaved 分块)。
2. 描述 TP 列/行并行与双 AllReduce: 列并行输出为什么需要 AllReduce 求和、行并行反向为什么也需要; 每 transformer 层每迭代共几次(前向 2 + 反向 2 = 4)。
3. 对比 greedy 与 1F1B 调度: 为什么 1F1B 峰值显存更低(in-flight microbatch ≤ p)、反向启动更早; 气泡为何没有因此消除(stage 间的流水线等待)。
4. 推导 Ring-Attention: 序列切 N 段后每卡 KV 显存为何 O(1); 每层通信量 ≈ 2 × 全部 KV; 带宽/延迟特性(带宽恒定、N−1 次串行 hop)与 FlashAttention 的关系。
5. 3D 整合题: 按通信层次解释 DP/TP/PP 分工(TP 节点内高频小消息、PP 节点间低频大消息、DP 梯度 AllReduce), 为什么 3D 并行能同时扩大并行度、降低单卡显存与通信开销。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/megatron-3d-bubble?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=megatron-3d-bubble)

---

## 21. MLOps Pipeline · MLOps 训练管线

> the MLOps pipeline industrializes the model lifecycle “data → training → evaluation → deployment” with four pillars: ① CI/CD for ML — code, data and models are all artifacts with their own build/test/release pipelines, so a data change or code merge auto-trigg…

**Interview questions**

1. How does CI/CD for ML differ from traditional software CI/CD? Why must data and models be managed as artifacts?
2. Which fields must experiment tracking (MLflow/W&B) record at minimum, and how does it keep cross-experiment comparison fair?
3. How do you make training reproducible? Is fixing the seed enough (data version/env/framework/hardware)?
4. Model registry vs model repository? How to design canary traffic ramps and rollback?
5. Given a notebook training run, how does a colleague reproduce identical results on another machine? Where do you look when it fails?

<details>
<summary>中文题目</summary>

1. MLOps 的 CI/CD 和传统软件 CI/CD 有什么不同?为什么数据和模型也要当 artifact 管?
2. 实验追踪(MLflow/W&B)至少要记录哪些字段?怎样保证跨实验对比公平?
3. 训练可复现性怎么保证?只固定随机种子够吗?(数据版本/环境/框架版本/硬件)
4. 模型注册表与模型仓库有什么区别?灰度发布如何设计流量比例与回滚?
5. 给定一次 notebook 训练产出,如何让同事在另一台机器上复现出相同结果?从哪几方面排查不可复现?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/mlops-pipeline?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mlops-pipeline)

---

## 22. Model Risk Governance · 模型风险治理

> model risk governance targets three output risks of generative models: ① jailbreak & prompt injection — jailbreaks use crafted prompts (role-play, fictional scenarios, multilingual variants) to bypass safety alignment, while prompt injection smuggles malicious…

**Interview questions**

1. Jailbreak vs prompt injection — the difference? Typical attack patterns of each?
2. What ranks #1 in the OWASP LLM Top 10, and why does it outrank traditional application-layer vulnerabilities?
3. How is output abuse detected — harmful-content classifiers, PII re-scanning, link/code allowlists, and what each solves?
4. How to design red teaming? Which attack categories to cover, and how to quantify success rates and track fixes?
5. How to implement audit and traceability? Which log fields are needed to support accountability and regulatory evidence?

<details>
<summary>中文题目</summary>

1. 越狱(Jailbreak)和提示注入(Prompt Injection)的区别?各有什么典型攻击方式?
2. OWASP LLM Top 10 排名第一的风险是什么?为什么它排在应用层漏洞之上?
3. 输出滥用检测怎么做?有害内容分类器、PII 二次扫描、链接/代码白名单各自解决什么?
4. 红队测试如何设计?覆盖哪些攻击类别,成功率如何量化与追踪修复?
5. 审计与可追溯如何落地?日志需要记录哪些字段才能支撑事后追责与监管举证?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/model-risks?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=model-risks)

---

## 23. Model Testing & Gates · 模型测试与评估门禁

> model testing & gates extends “testing code” to “testing models”, in four layers: ① unit/integration tests — validating pure functions and pipeline components (data transforms, tokenizers, post-processing) so the code logic is correct;

**Interview questions**

1. How does model testing differ from traditional software testing? What do unit vs integration tests cover in a model pipeline?
2. How to design an eval gate? How are thresholds and eval sets chosen without overfitting the gate?
3. What is the regression baseline for? What if a new model passes the overall gate but regresses on a subset (long text, low-resource language)?
4. What goes into the pre-deployment checklist? How do you self-check for data leakage and train/eval contamination?
5. Why does an eval gate fail when the eval set matches the training distribution? How to build a leaderboard-resistant eval set?

<details>
<summary>中文题目</summary>

1. 模型测试和传统软件测试有什么不同?单元/集成测试分别覆盖模型链路里的什么?
2. 评估门禁怎么设计?阈值怎么定、评估集怎么选才不会把模型“训到门禁上”?
3. 回归基线的意义?新模型总指标达标但某个子集(如长文本/低资源语言)掉分怎么办?
4. 上线前检查清单包含哪些项?如何自查数据泄漏与训练/评估集交叉污染?
5. 为什么评估集与训练集同分布时门禁会失真?如何构建抗“刷榜”的评估集?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/model-testing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=model-testing)

---

## 24. Model Weight & Checkpoint Storage · 模型权重与检查点存储

> Model weight and checkpoint storage. Safetensors separates tensor data from layout metadata: a JSON header describes each tensor's name, shape, dtype and byte offset, followed by the raw tensor bytes, enabling mmap zero-copy loading;

**Interview questions**

1. Safetensors vs pickle? Why is unpickling model weights unsafe?
2. Why does Safetensors load faster? The mechanics of mmap zero-copy and lazy loading?
3. Single-file vs sharded checkpoints? How does an index enable parallel load on resume?
4. Core capabilities of a model registry (HF Hub style)? What should a Model Card contain?
5. How to design production model rollback? Immutable versions, atomic publish and canary?

<details>
<summary>中文题目</summary>

1. Safetensors 与 pickle 的区别? 为什么用 pickle 反序列化模型权重不安全?
2. Safetensors 为什么加载更快? mmap 零拷贝与惰性加载的原理是什么?
3. 大模型检查点单文件 vs 分片 (sharded) 的取舍? 恢复时如何用 index 并行加载?
4. 模型仓库 (Model Registry / HF Hub 式) 的核心能力? Model Card 应包含什么?
5. 线上模型版本回滚如何设计? 不可变版本、原子发布与灰度如何配合?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/model-storage?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=model-storage)

---

## 25. Monitoring & Observability · 监控与可观测性

> LLM serving monitoring has three layers: ① system-level metrics — latency (P50/P95/P99), throughput (QPS, tokens/sec), error rate (4xx/5xx and timeouts), GPU utilization, KV-cache hit rate and queue depth;

**Interview questions**

1. What do TTFT and TPOT measure and how do they interact? How is $T = TTFT + TPOT \times (N-1)$ used?
2. Why track P99 latency instead of the average? Where do tail requests come from in LLM serving?
3. Under a QPS spike, how do traces separate input-side, serving-side and model-side bottlenecks?
4. How to set SLOs and alerts? How to design thresholds and severity tiers for error rate and latency?
5. How do serving-side metrics — KV-cache hit rate, GPU utilization, queue depth — drive scaling decisions?

<details>
<summary>中文题目</summary>

1. TTFT 和 TPOT 各衡量什么、互相怎么影响?总延迟公式 $T = TTFT + TPOT \times (N-1)$ 如何使用?
2. 为什么监控延迟要看 P99 而不是平均值?LLM 服务的长尾请求来自哪里?
3. QPS 突增时,如何用 trace 区分输入侧/服务侧/模型侧瓶颈?
4. SLO 与告警怎么设?错误率、延迟告警的阈值与分级如何设计?
5. KV cache 命中率、GPU 利用率、排队长度这些服务侧指标如何指导扩缩容?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/monitoring-observability?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=monitoring-observability)

---

## 26. Prefill/Decode Disaggregation · Prefill/Decode 分离

> Prefill/Decode disaggregation (PD split) = running the two phases of a request on different instance types: Prefill processes the whole prompt in parallel — matrix-matrix (GEMM) work that is compute-bound with high GPU utilization;

**Interview questions**

1. Why is Prefill compute-bound while Decode is bandwidth-bound (GEMM vs GEMV — reading all weights per step to emit one token)? Explain with a Roofline model.
2. PD split architecture: how does the KV cache travel from Prefill to Decode instances (network / async pipelining); what new bottleneck does the split introduce?
3. Why is Decode-instance load balancing based on occupied KV size / sequence length instead of connection or request counts? How do you avoid imbalance with mixed long and short requests?
4. Which phase decides TTFT vs ITL (TPOT)? How does PD split let you optimize them independently, and why do they fight each other when colocated?
5. Relation between PD split, continuous batching and chunked prefill: why is chunked prefill a "soft disaggregation" alternative, and what are the trade-offs of each?

<details>
<summary>中文题目</summary>

1. 为什么 Prefill 是计算密集而 Decode 是带宽密集(GEMM vs GEMV,每步读全量权重只产 1 个 token)?用 Roofline 模型解释两者的瓶颈不同。
2. PD 分离的架构: KV Cache 如何从 Prefill 实例传到 Decode 实例(网络/异步管线化);分离后引入的新瓶颈是什么?
3. Decode 实例的负载均衡为什么按已占 KV 大小/序列长度,而不是连接数或请求数?长短请求混合时如何避免不平衡?
4. TTFT 与 ITL(TPOT)分别由哪个阶段决定?PD 分离如何让两者独立优化,合部部署时为何互相牵制?
5. PD 分离与连续批处理、chunked prefill 的关系: chunked prefill 为何是“软分离”的替代方案,各自取舍是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/prefill-decode-split?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=prefill-decode-split)

---

## 27. Prompt Caching & RadixAttention · Prompt Caching/RadixAttention

> Prompt caching reuses the KV computed for identical prompt prefixes, skipping repeated prefill computation. Typical reusable prefixes: system prompts, few-shot exemplars, early turns of multi-turn conversations, and shared instruction blocks across tenants — o…

**Interview questions**

1. Radix-tree prefix reuse: how a new request does longest-prefix matching; how hit prefixes vs unmatched suffixes are handled; how the tree splits/merges as requests end.
2. Why TTFT drops 80%+: prefill compute is proportional to unmatched tokens; convert hit rate and prefill share into numbers (prefill is 90% of TTFT, 90% hits → TTFT down ~81%).
3. Eviction policies: how LRU / reference counting work on the radix tree (branch-wise eviction); the cost of evicting hot prefixes on hit rate and TTFT.
4. Which workloads gain most: system prompts, multi-turn, multi-tenant sharing; estimate hit rate and token savings, and how caching composes with continuous batching and PD disaggregation.
5. vLLM prefix caching vs SGLang RadixAttention: block-hash matching vs radix-tree longest-prefix matching; hit granularity, implementation complexity, and the gap on long shared prefixes.

<details>
<summary>中文题目</summary>

1. RadixTree 前缀复用原理: 新请求如何做最长前缀匹配;命中的前缀与未命中的后缀分别怎么处理;请求结束时树如何分裂/合并。
2. 为什么 TTFT 能降 80%+: prefill 计算量正比于未命中 token 数;给出命中率与 prefill 占比的数值换算(prefill 占 TTFT 90% 时,90% 命中 → TTFT 约降 81%)。
3. 缓存淘汰策略: LRU / 引用计数在基数树上怎么实现(逐分支淘汰);淘汰错误(驱逐热点前缀)对 hit rate 和 TTFT 的代价。
4. 哪些场景收益最大: 系统提示、多轮对话、多租户共享;hint: 命中率与 token 节省如何估算,与连续批/PD 分离如何叠加。
5. vLLM prefix caching vs SGLang RadixAttention: block 哈希匹配 vs 基数树最长前缀匹配;命中粒度、实现复杂度、长共享前缀场景差异。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/prompt-caching-radix?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=prompt-caching-radix)

---

## 28. RL Training/Serving (cross-module) · RL 训练与推理服务(跨模块)

> RL training/serving is the sampling-and-verification infrastructure for RLVR (RL with Verifiable Rewards): training samples $N$ rollout trajectories per prompt and scores/filters them with a verifier; serving hosts test-time scaling through extended generation.

**Interview questions**

1. RLVR vs RLHF (verifiable rewards vs reward models) — how to design the sampling-verification pipeline?
2. Large-scale rollout infra: how to schedule training/serving co-location, sampling throughput, and verifier compute?
3. How are verifiers built to score/filter trajectories (answer checking, unit-test execution)?
4. Test-time scaling: how do longer generation or repeated sampling improve accuracy, and how is cost controlled?
5. How is rollout data replayed into policy updates (GRPO/PPO), and how does sampling parallelism affect reward distribution?

<details>
<summary>中文题目</summary>

1. RLVR 与 RLHF 的差异 (可验证奖励 vs 奖励模型), 采样-验证管线如何设计?
2. 大规模 rollout 的基础设施: 训练/推理混部、采样吞吐、验证器计算如何调度?
3. 验证器 (verifier) 如何构造并过滤轨迹 (答案比对、单测执行)?
4. 推理时扩展 (test-time scaling): 延长生成/多次采样如何提升准确率, 成本如何控制?
5. rollout 数据如何回放给策略更新 (GRPO/PPO), 采样并行度如何影响奖励分布?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/rl-serving?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rl-serving)

---

## 29. Security & Privacy · 安全与隐私

> four security & privacy problems for LLM services: ① model extraction (distillation attack) defense — adversaries distill a near-equivalent model through API queries;

**Interview questions**

1. How does distillation (model extraction) work, and what API-side defenses exist (detection, watermarking, rate limiting)?
2. How does data poisoning happen, how is it caught pre-training, and how do you spot backdoor behavior post-deployment?
3. Where should PII redaction sit in the pipeline? Why both input and output sanitization?
4. What does GDPR require for training and inference data? How do you assess cross-border data transfers?
5. Intuition and formula of differential privacy? Is smaller or larger ε better, and how do you trade it against utility?

<details>
<summary>中文题目</summary>

1. 蒸馏攻击/模型窃取的原理?API 侧有哪些防护手段(检测/水印/限流)?
2. 数据投毒如何发生、训练前怎么检测、上线后如何发现后门行为?
3. PII 脱敏放在管线哪个位置?输入脱敏与输出脱敏为什么都要?
4. GDPR 对训练数据与推理数据的要求?数据出境如何进行合规评估?
5. 差分隐私的直觉与公式 $\Pr[\mathcal{M}(D) \in S] \le e^{\epsilon} \Pr[\mathcal{M}(D') \in S]$?ε 越小越好还是越大越好,怎么权衡效用?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/security-privacy?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=security-privacy)

---

## 30. Speculative Decoding · 推测解码

> Speculative decoding uses a small, fast draft model to guess γ candidate tokens ahead, then has the target model verify the whole candidate sequence in one parallel forward pass, accepting matching tokens and resampling where they diverge — turning each forwar…

**Interview questions**

1. Describe the draft-verify flow and hand-compute the speedup: $E \approx \frac{1 - \alpha^{\gamma}}{1 - \alpha}$; with $\alpha=0.8,\gamma=4$ get ≈ 2.95×; explain why larger $\gamma$ has diminishing returns (bounded by $1/(1-\alpha)$).
2. Why verifying γ candidates in one pass costs almost no extra latency: candidates enter as one forward, per-position logits computed in parallel; contrast serial 1-token decode with γ-token verification.
3. How rejection sampling provably matches the target distribution (lossless): the math behind accept probability $\min(1, p_i/q_i)$ and resampling from $\text{norm}(\max(0, p_i - q_i))$.
4. EAGLE vs Medusa vs a separate small draft model: how drafts are produced, acceptance quality, training requirements, and how tree attention verifies multiple branches at once.
5. What determines the acceptance rate $\alpha$, how to measure and improve it; how draft-model distillation or weight sharing affects throughput and memory.

<details>
<summary>中文题目</summary>

1. 写出推测解码流程并手算加速比: 公式 $E \approx \frac{1 - \alpha^{\gamma}}{1 - \alpha}$;给 $\alpha=0.8,\gamma=4$ 算出 ≈ 2.95×;解释为什么 $\gamma$ 增大收益递减(上限 $1/(1-\alpha)$)。
2. 为什么“一次验证 γ 个候选”几乎不增加延迟: 候选 token 单次前向输入、按位置并行计算 logits;比较 decode 串行 1 token 与验证 γ token 的计算特性。
3. rejection sampling 如何保证输出分布与目标模型一致(lossless): 接受概率 $\min(1, p_i/q_i)$ 与重采样 $\text{norm}(\max(0, p_i - q_i))$ 的数学含义。
4. EAGLE vs Medusa vs 独立小草稿模型: 各自草稿怎么来、接受率高低、是否需要额外训练、树状验证 (tree attention) 如何一次验证多分支。
5. 接受率 $\alpha$ 受什么影响、如何测量与提升;草稿模型与小模型的蒸馏/共享权重策略对吞吐与显存的影响。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/speculative-decoding?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=speculative-decoding)

---

## 31. Training Frameworks & Profiling · 训练框架与 Profiling

> A training framework is the combination of parallel strategy + memory optimization + observability. PyTorch native offers two paths: DDP (data parallelism;

**Interview questions**

1. Compare the memory budgets: DDP holds $16\Psi$ per rank (full replica, ZeRO-0), FSDP/ZeRO-3 hold $16\Psi/N_d$, DeepSpeed adds CPU/NVMe offload; state which solves compute scaling and which solves memory scaling.
2. Explain FSDP sharding: weight tensors sharded by rank → All-Gather the layer's params before forward → discard after use → Reduce-Scatter gradient shards in backward; why communication is ~1.5× DDP (3Φ vs 2Φ) and why scaling stays near-linear.
3. Selection: FSDP for single-node multi-GPU with minimal code; DeepSpeed for cross-node, offload, very large scale, or its ecosystem; what differs between FSDP's FULL_SHARD and SHARD_GRAD_OP.
4. Use torch.profiler to find bottlenecks: split a step into compute/comm/IO/idle; how to fix un-overlapped communication (bucket size, async gradients) and IO stalls (worker count, prefetch_factor, data format); give a debug order.
5. Worked scenario: why DDP cannot fit 70B on 8×80 GB ($16\Psi = 1120$ GB > 640 GB); FSDP/ZeRO-3 gives $1120/8 = 140$ GB per rank — still over 80 GB, so 16 ranks or offload ($2\Psi/N_d = 17.5$ GB on GPU) is required; give the full two-level accounting.

<details>
<summary>中文题目</summary>

1. 对比三者显存账: DDP 每卡 $16\Psi$(全量副本,ZeRO-0),FSDP/ZeRO-3 每卡 $16\Psi/N_d$,DeepSpeed 额外支持 CPU/NVMe Offload;分别说明解决的是算力扩展还是显存扩展。
2. 讲清 FSDP 分片原理: 权重张量按 rank 分片 → 前向 All-Gather 恢复本层参数 → 用后即弃 → 反向 Reduce-Scatter 梯度分片;为什么通信量约 DDP 的 1.5 倍(3Φ vs 2Φ)、为什么仍近线性扩展。
3. 选型: 单机多卡、想少写代码选什么(FSDP);跨机/需要 Offload/超大规模/既有生态选什么(DeepSpeed);FSDP 的 FULL_SHARD 与 SHARD_GRAD_OP 有何区别。
4. 用 torch.profiler 定位瓶颈: 把单步时间拆成 compute/comm/IO/idle 四类;通信未与反向重叠怎么调(bucket 大小、异步梯度),IO 打不满怎么调(worker 数、prefetch_factor、数据格式),给出排查顺序。
5. 手算场景: 70B 在 8×80 GB 节点上为什么 DDP 装不下($16\Psi = 1120$ GB > 640 GB);FSDP/ZeRO-3 每卡 $1120/8 = 140$ GB 仍超 80 GB,须 16 卡或再开 Offload(GPU 仅剩 $2\Psi/N_d = 17.5$ GB);完整说出两层账。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/training-frameworks?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=training-frameworks)

---

## 32. ZeRO-1/2/3 & Offload · ZeRO-1/2/3 显存切分

> ZeRO (Zero Redundancy Optimizer, DeepSpeed) fits large-model training into limited GPU memory by eliminating redundant copies of state.

**Interview questions**

1. By hand, compute the per-rank static memory for training a 70B model (Adam + FP16/FP32 mixed precision): write $16\Psi = 2\Psi + 2\Psi + 12\Psi$, name each term (FP16 params / FP16 grads / FP32 master weights + first + second moments), arrive at 1120 GB, and explain why naive DDP still costs 1120 GB per rank (N redundant copies).
2. Write the per-rank memory formulas for ZeRO-1/2/3: $2\Psi + 2\Psi + \tfrac{12\Psi}{N_d}$, $2\Psi + \tfrac{2\Psi + 12\Psi}{N_d}$, $\tfrac{16\Psi}{N_d}$; state what each stage shards and what remains replicated.
3. What does ZeRO-2 save over ZeRO-1, and by how much (gradients drop from $2\Psi$ to $2\Psi/N_d$ per rank)? Why can gradients be sharded — Reduce-Scatter right after backprop instead of a full All-Reduce.
4. Why is ZeRO-3's communication ~1.5× DDP (forward All-Gather Φ + backward Reduce-Scatter Φ + a second All-Gather for recompute Φ ≈ 3Φ, vs DDP's 2Φ), and is the memory win $16\Psi \to 16\Psi/N_d$ worth it?
5. What does ZeRO-Offload move to the CPU and at what cost: optimizer states and the Adam step run on the CPU while the GPU keeps only the layer's param shard $2\Psi/N_d$; memory drops dramatically at the price of PCIe-bound, slower steps; how does ZeRO-Infinity extend to NVMe?

<details>
<summary>中文题目</summary>

1. 手算 70B 模型(Adam + FP16/FP32 混合精度)训练的单卡静态显存: 写出 $16\Psi = 2\Psi + 2\Psi + 12\Psi$,逐项说明(FP16 参数 / FP16 梯度 / FP32 主权重 + 一阶 + 二阶动量),得出 1120 GB,并解释为何朴素 DDP 每卡仍是 1120 GB(复制 N 份冗余)。
2. 写出 ZeRO-1/2/3 三档的每卡显存公式 $2\Psi + 2\Psi + \tfrac{12\Psi}{N_d}$、$2\Psi + \tfrac{2\Psi + 12\Psi}{N_d}$、$\tfrac{16\Psi}{N_d}$,并说明每一档切了什么、还剩什么未切。
3. ZeRO-2 相比 ZeRO-1 省了什么?省多少(梯度从每卡 $2\Psi$ 降到 $2\Psi/N_d$)?梯度为什么能被切——反向后立即 Reduce-Scatter 聚合分片,而不是全量 All-Reduce。
4. ZeRO-3 的通信量为什么约是 DDP 的 1.5 倍(前向 All-Gather 参数 Φ + 反向 Reduce-Scatter Φ + 反向重算时再 All-Gather 参数 Φ ≈ 3Φ,vs DDP 的 2Φ),显存收益 $16\Psi \to 16\Psi/N_d$ 是否划算。
5. ZeRO-Offload 把什么移到 CPU、代价是什么: 优化器状态与 Adam 计算下放 CPU,GPU 只留本层参数分片 $2\Psi/N_d$;换来显存大降,代价是 PCIe 带宽受限、步时间变长;ZeRO-Infinity 如何再扩展到 NVMe。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Infra/card/zero-memory-offload?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=zero-memory-offload)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
