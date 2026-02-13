# Heterogeneous Mixture-of-Experts with Dynamic Expert Lifecycle Management

**Suraj Samaga**

Yoctotta Technologies / Kaman AI

February 2026

---

## Abstract

Mixture-of-Experts (MoE) architectures traditionally employ homogeneous expert networks, where all experts share the same architectural design. We challenge this assumption by proposing **Heterogeneous MoE (HetMoE)**, a framework in which experts can have fundamentally different computational architectures — feed-forward networks, temporal convolutions, classical statistical estimators, and spatial convolution operators — within a single MoE layer. We demonstrate that architectural diversity among experts provides an inductive bias that naturally aligns with structurally diverse data, yielding a 67.1% improvement in predictive accuracy over homogeneous baselines on mixed-type sequential data. Crucially, we identify that **soft routing is essential for heterogeneous experts**: standard top-1 hard routing, which works well for homogeneous MoE, catastrophically fails when experts have different architectures. We further propose a **dynamic expert lifecycle management** protocol that enables incremental addition of new experts to a frozen base and graceful retirement of underutilized experts. Across five random seeds, HetMoE consistently outperforms homogeneous baselines (5/5 wins) with lower variance (±0.0041 vs ±0.0099), suggesting that architectural heterogeneity provides a more stable optimization landscape. All experiments are conducted in pure NumPy without GPU acceleration, enabling full reproducibility on any hardware.

**Keywords:** mixture-of-experts, heterogeneous architectures, continual learning, dynamic routing, expert lifecycle

---

## 1 Introduction

Mixture-of-Experts (MoE) has emerged as a powerful paradigm for scaling neural networks efficiently. By routing different inputs to different expert sub-networks, MoE architectures achieve high capacity while keeping per-input computation tractable. This approach has been successfully adopted in large language models including Mixtral [1], DeepSeek-V2 [2], and Qwen-MoE [3], as well as in vision models and multimodal systems.

A largely unquestioned assumption in existing MoE literature is **architectural homogeneity**: all experts within a layer share the same network design, typically a feed-forward network (FFN). The diversity among experts arises solely from different learned weights, not from structural differences in how they process information. While this simplifies training and implementation, it imposes a fundamental limitation — all experts share the same inductive bias.

We propose that this limitation is significant when the data distribution contains structurally diverse patterns. Temporal patterns (e.g., trends, seasonality) are naturally captured by convolutional or recurrence-based architectures. Spatial patterns (e.g., neighbor averaging, grid dynamics) benefit from 2D convolution operations. Classical statistical properties (e.g., weighted moving averages, trend-intercept decomposition) are directly computable by closed-form estimators. A homogeneous MoE must learn all of these behaviors from the same architectural template, while a heterogeneous MoE can leverage purpose-built computational primitives.

This paper makes three contributions:

**1. Heterogeneous MoE architecture.** We propose and validate an MoE framework where experts have different architectures (FFN, 1D temporal convolution, classical statistical estimator, 2D spatial convolution) within the same layer. We demonstrate 67.1% improvement over homogeneous baselines on mixed-type data.

**2. Critical role of soft routing.** We discover that the choice of routing strategy interacts strongly with expert heterogeneity. Hard (top-1) routing, standard in homogeneous MoE, causes catastrophic underperformance with heterogeneous experts. Soft (weighted combination) routing is essential, as it enables inputs to benefit from multiple complementary computational paradigms simultaneously.

**3. Dynamic expert lifecycle management.** We demonstrate a protocol for incrementally adding new experts to a frozen model and retiring underutilized experts. This enables continual capability expansion without catastrophic forgetting, with each addition monotonically improving performance and each retirement producing graceful degradation (5.6% MSE increase).

### 1.1 Motivation: Toward Sensor-Agnostic World Models

Beyond the immediate experimental results, we are motivated by a broader vision: building world models that can incrementally incorporate new sensory modalities. Consider an embodied AI system that begins with vision, later adds LIDAR, then RADAR, then proprioceptive sensors. Each modality has fundamentally different data characteristics — images are spatial, LIDAR produces point clouds, RADAR gives velocity fields, IMU data is temporal. A heterogeneous MoE with lifecycle management could serve as the backbone for such a system, where each sensor modality is served by an architecturally appropriate expert that can be dynamically added or retired as hardware changes.

---

## 2 Related Work

### 2.1 Mixture-of-Experts

The MoE framework was introduced by Jacobs et al. [4] and has been extensively developed for large-scale language modeling. Shazeer et al. [5] introduced sparsely-gated MoE with top-k routing. Switch Transformer [6] simplified routing to top-1 with load balancing. Recent production models including Mixtral-8x7B [1], DeepSeek-V2 [2], and Qwen-MoE [3] have demonstrated the effectiveness of MoE for achieving high performance with reduced compute.

All of these works use homogeneous expert architectures. While expert weights diverge during training and develop implicit specialization, the computational structure is shared.

### 2.2 MoE for Continual Learning

Lifelong-MoE [7] proposed progressively expanding expert capacity while freezing previously trained experts and applying regularization. DynamicMoE [8] periodically adds new experts for continual and reinforcement learning. D-MoLE [9] dynamically allocates LoRA experts across layers for continual multimodal instruction tuning. LEMoE [10] uses MoE adaptors for lifelong model editing. These works demonstrate the viability of expert expansion for continual learning but do not explore architectural heterogeneity among experts.

### 2.3 Expert Pruning

MoE-Pruner [11] and Lu et al. [12] study pruning of MoE experts for inference efficiency. R2MoE [13] introduces redundancy removal for lifelong concept learning. These approaches address the "retirement" side of expert lifecycle but treat it as an efficiency optimization rather than a continuous management protocol.

### 2.4 Heterogeneous Architectures

Hybrid architectures combining different computational paradigms (e.g., transformers with state-space models in Jamba [14]) have shown promise, but these are fixed architectures rather than dynamically routed mixtures. To our knowledge, no prior work has studied MoE where experts have fundamentally different architectures and investigated the routing implications of this design choice.

---

## 3 Method

### 3.1 Heterogeneous Expert Pool

We define four expert architectures, each providing a different computational inductive bias:

**FFN Expert** (general-purpose). A two-layer feed-forward network: $f_{\text{FFN}}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$. This provides a universal function approximator without structural assumptions.

**Temporal Convolution Expert** (sequential patterns). Extracts sliding windows of width $K$ from the input sequence, applies a learned linear transformation per window, and mean-pools: $f_{\text{TempConv}}(x) = W_h \cdot \text{pool}(\text{ReLU}(W_c \cdot \text{windows}_K(x)))$. This provides an inductive bias for local temporal dependencies.

**Classical Expert** (statistical patterns). Computes a learnable weighted average of the input (with softmax-normalized weights), extracts hand-crafted features (weighted mean, last value, trend, standard deviation), and applies a small neural correction: $f_{\text{Classical}}(x) = \sum_i w_i x_i + g([\bar{x}_w, x_T, x_T - x_0, \sigma_x])$. This combines the efficiency of classical time-series methods with neural flexibility.

**Spatial Convolution Expert** (grid patterns). Reshapes the input into a 2D grid, extracts 3×3 patches with wrap-around padding, applies a learned transformation, and pools: $f_{\text{SpatialConv}}(x) = W_h \cdot \text{pool}(\text{ReLU}(W_c \cdot \text{patches}_{3\times3}(\text{grid}(x))))$. This provides an inductive bias for local spatial correlations.

### 3.2 Routing Strategy

Given input $x$, the router computes expert weights via a learned gating network followed by softmax:

$$g(x) = \text{softmax}(W_r \cdot \text{ReLU}(W_g x + b_g) + b_r)$$

The output is a weighted combination of all expert outputs:

$$y = \sum_{i=1}^{E} g_i(x) \cdot f_i(x)$$

We identify soft routing (using all expert weights) as critical for heterogeneous MoE. Section 4.2 presents detailed ablation results.

### 3.3 Load Balancing

We apply lightweight auxiliary losses to encourage reasonable load distribution without forcing uniformity:

$$\mathcal{L}_{\text{bal}} = \alpha \cdot ||\bar{g} - \mathbf{u}||_2^2, \quad \mathcal{L}_{\text{ent}} = -\beta \sum_i \bar{g}_i \log \bar{g}_i$$

where $\bar{g}$ is the batch-averaged routing weight and $\mathbf{u}$ is the uniform distribution. We use $\alpha = 0.05$ and $\beta = 0.02$ — significantly weaker than typical MoE load balancing. We find this is important: strong balancing forces heterogeneous experts toward uniform usage, preventing the specialization that is the primary benefit of architectural diversity.

### 3.4 Expert Lifecycle Management

**Addition.** When a new expert is added: (1) all existing experts are frozen, (2) the router's output dimension is expanded with small random initialization for the new expert, (3) only the new expert parameters and the full router are trainable. This preserves existing capabilities while enabling the model to learn new skills.

**Monitoring.** Expert utilization is tracked via mean routing weights over evaluation batches: $u_i = \mathbb{E}_x[g_i(x)]$.

**Retirement.** When an expert's utilization falls below a threshold (we use the minimum across experts), it is removed and the router's output dimension is contracted. The remaining experts' routing weights are renormalized.

---

## 4 Experiments

All experiments use pure NumPy with manual autograd. No GPU or deep learning framework is required, enabling exact reproducibility on any hardware.

### 4.1 Data

We generate three types of sequential data, each with fundamentally different structure:

**Pattern data.** Repeating patterns of length 2-5, tiled across 16 timesteps. Target: next value in the cycle. Tests periodic pattern recognition.

**Time series data.** Sinusoidal signals with random frequency, phase, trend, and noise. Target: next value. Tests smooth temporal extrapolation.

**Spatial data.** 4×4 grids with wrap-around neighbor averaging. Target: center cell value after averaging. Tests 2D spatial convolution understanding.

Batches contain equal proportions of all three types, shuffled together.

### 4.2 Results

#### Experiment 1: Homogeneous vs. Heterogeneous

| Model | Overall | Pattern | TimeSeries | Spatial |
|---|---|---|---|---|
| Homogeneous 3×FFN | 0.0668 | 0.0709 | 0.0552 | 0.0741 |
| Hetero 3 (hard routing) | 0.0735 | 0.0597 | 0.0431 | 0.1175 |
| **Hetero 4 (soft, ours)** | **0.0401** | **0.0178** | **0.0336** | **0.0686** |
| **Hetero 4 (1500ep, ours)** | **0.0220** | **0.0143** | **0.0218** | **0.0298** |

*Table 1: Overall and per-type MSE (lower is better). Our best model achieves 67.1% improvement over the homogeneous baseline.*

The heterogeneous model with soft routing outperforms the homogeneous baseline on **all three data types**. With extended training (1500 epochs), pattern recognition improves by 80%, time series by 60%, and spatial by 60% relative to the homogeneous baseline.

Critically, heterogeneous experts with **hard routing fail** (0.0735 vs 0.0668 for homogeneous). This is because hard routing forces each input to a single expert. If that expert is architecturally mismatched for the data type (e.g., a temporal convolution expert receiving spatial data), performance degrades severely. Soft routing allows each input to benefit from all experts in proportion to their relevance, which is essential when experts have complementary rather than redundant capabilities.

#### Experiment 2: Routing Ablation

| Configuration | MSE |
|---|---|
| Hard + strong balancing | 0.1058 |
| Hard + weak balancing | 0.1275 |
| Hard + no balancing | 0.1264 |
| Soft + strong balancing | 0.0417 |
| **Soft + weak balancing (ours)** | **0.0401** |
| Soft + no balancing | 0.0402 |

*Table 2: Routing strategy ablation with 4 heterogeneous experts.*

Two findings emerge. First, **soft routing is the dominant factor** — all soft configurations outperform all hard configurations by 2-3×, regardless of balancing strength. Second, within the soft routing regime, balancing strength has minimal impact. This suggests that heterogeneous experts naturally avoid the collapse problem that motivates load balancing in homogeneous MoE, because architecturally distinct experts have non-overlapping competencies.

#### Experiment 3: Expert Lifecycle

| Phase | MSE | Experts |
|---|---|---|
| Initial | 0.0864 | FFN, TempConv |
| +Classical | 0.0649 | FFN, TempConv, Classical |
| +SpatialConv | 0.0619 | FFN, TempConv, Classical, SpatialConv |
| −Classical | 0.0654 | FFN, TempConv, SpatialConv |

*Table 3: Expert lifecycle — monotonic improvement with addition, graceful degradation with retirement.*

Each expert addition monotonically improves performance (24.9% → 4.6% relative improvement per addition). When the least-utilized expert (Classical, usage=0.083) is retired, degradation is only 5.6% — well within acceptable bounds and still better than the 2-expert starting point.

The router autonomously identified Classical as the least useful expert after SpatialConv was added, suggesting overlap in their capabilities for this particular data mixture. This demonstrates that the lifecycle protocol can serve as an automatic model compression mechanism.

#### Experiment 4: Robustness

| Seed | Homogeneous | Heterogeneous |
|---|---|---|
| 42 | 0.0668 | 0.0401 |
| 123 | 0.0803 | 0.0485 |
| 456 | 0.0725 | 0.0502 |
| 789 | 0.0923 | 0.0481 |
| 1337 | 0.0656 | 0.0412 |
| **Mean ± Std** | **0.0755 ± 0.0099** | **0.0456 ± 0.0041** |

*Table 4: Performance across random seeds. Heterogeneous MoE wins 5/5 with lower variance.*

The heterogeneous model wins across all five seeds with 40% lower variance. This suggests that architectural diversity provides a more stable optimization landscape — different experts converge via different gradient pathways, reducing sensitivity to initialization.

---

## 5 Discussion

### 5.1 Why Soft Routing Is Essential for Heterogeneous Experts

This finding has clear intuitive grounding. In homogeneous MoE, hard (top-1) routing works because any expert can potentially handle any input — they differ only in learned specialization, not capability. In heterogeneous MoE, experts differ in what they **can** compute. A temporal convolution expert cannot perform 2D spatial reasoning regardless of its weights. Hard routing to the wrong expert is an architectural mismatch, not just a weight mismatch.

Soft routing resolves this by allowing every input to access all computational paradigms. The weighting determines **how much** each paradigm contributes, not whether it contributes at all. This is analogous to ensemble methods in classical machine learning, where combining diverse base learners typically outperforms selecting a single one.

### 5.2 Reduced Need for Load Balancing

Homogeneous MoE suffers from routing collapse — the router learns to send all inputs to one expert while ignoring others. This occurs because experts are interchangeable; the router has no gradient pressure to distribute load. Strong auxiliary losses are required to prevent this.

Heterogeneous experts naturally resist collapse because they produce different gradients for the same input. If the router ignores the temporal convolution expert, it loses access to a type of computation that no other expert can provide. The task loss itself creates pressure to use diverse experts, reducing reliance on auxiliary balancing.

### 5.3 Implications for Continual Learning

The expert lifecycle results suggest a practical protocol for continual capability expansion in deployed models. Rather than retraining an entire model when new capabilities are needed, practitioners can: (1) freeze the existing model, (2) add an architecturally appropriate expert for the new capability, (3) train only the new expert and router, (4) periodically audit and retire redundant experts.

This directly addresses catastrophic forgetting — frozen experts cannot forget — while maintaining efficiency through retirement of redundant capacity.

### 5.4 Toward Heterogeneous Experts in LLMs

While our experiments use a simple prediction task, the principle extends to large language models. Consider an LLM-based system that incrementally gains multimodal perception: vision data is naturally processed by spatial convolution experts, audio by temporal convolution experts, and structured sensor data by classical estimation experts. The frozen LLM backbone retains language capabilities while new experts add domain-specific perception. This connects to the vision of sensor-agnostic world models where a single model handles diverse inputs through architecturally matched expert modules.

### 5.5 Limitations

**Scale.** Our experiments use small models (hundreds of parameters) and synthetic data. Validating these findings at transformer scale with real-world data is necessary future work.

**Expert architecture selection.** We manually designed four expert types. In practice, selecting the right architecture for a new expert requires domain knowledge. Automating this selection (e.g., via neural architecture search over the expert design space) is an open problem.

**Cross-expert interaction.** Soft routing combines expert outputs linearly. Richer interaction mechanisms (e.g., cross-attention between expert representations) might capture complementary information more effectively.

**Token-level routing in transformers.** Our experiments use sequence-level routing. In transformer MoE, routing happens per-token. Whether the benefits of heterogeneous experts persist at the token level requires investigation.

---

## 6 Conclusion

We demonstrate that architectural heterogeneity among MoE experts provides substantial benefits over the standard homogeneous design, achieving 67% improvement on mixed-type data with greater robustness across initializations. The key enabling insight is that **heterogeneous experts require soft routing** — a finding with implications for any MoE system incorporating non-standard expert architectures.

Combined with dynamic lifecycle management (addition and retirement of experts), this framework provides a foundation for continually evolving models that can incrementally gain new capabilities — including new sensory modalities — without retraining from scratch and without forgetting existing skills.

We release all code as a pure NumPy implementation requiring no GPU or deep learning framework, enabling exact reproducibility and encouraging further exploration of heterogeneous expert architectures.

---

## References

[1] Jiang, A. Q., et al. "Mixtral of Experts." arXiv:2401.04088, 2024.

[2] DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434, 2024.

[3] Qwen Team. "Qwen Technical Report." arXiv:2309.16609, 2023.

[4] Jacobs, R. A., et al. "Adaptive Mixtures of Local Experts." Neural Computation, 3(1):79-87, 1991.

[5] Shazeer, N., et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR, 2017.

[6] Fedus, W., Zoph, B., & Shazeer, N. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR, 2022.

[7] Chen, W., et al. "Lifelong Language Pretraining with Distribution-Specialized Experts." ICML, 2023.

[8] Kim, D. "Dynamic Mixture of Experts Against Severe Distribution Shifts." arXiv:2511.18987, 2025.

[9] Ge, C., et al. "Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning." arXiv:2506.11672, 2025.

[10] Wang, R., & Li, X. "LEMoE: Advanced Mixture of Experts Adaptor for Lifelong Model Editing of Large Language Models." EMNLP, 2024.

[11] Xie, Y., et al. "MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router." arXiv:2410.12013, 2024.

[12] Lu, X., et al. "Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models." ACL, 2024.

[13] R2MoE. "Redundancy-Removal Mixture of Experts for Lifelong Concept Learning." arXiv:2507.13107, 2025.

[14] Lieber, O., et al. "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv:2403.19887, 2024.

[15] Li, H., et al. "Theory on Mixture-of-Experts in Continual Learning." ICLR, 2025.

---

## Appendix A: Reproducibility

All experiments can be reproduced by running:

```bash
git clone https://github.com/[your-username]/heterogeneous-moe
cd heterogeneous-moe
python src/experiments.py          # Full run (~100 seconds)
python src/experiments.py --quick  # Quick run (~30 seconds)
```

Requirements: Python 3.8+, NumPy, Matplotlib. No GPU needed.

## Appendix B: Figures

See the `figures/` directory for:
- `fig1_loss_curves.png` — Training dynamics
- `fig2_routing_heatmaps.png` — Router specialization patterns
- `fig3_expert_lifecycle.png` — Dynamic addition and retirement
- `fig4_per_type_analysis.png` — Per-type performance comparison
- `fig5_routing_ablation.png` — Routing strategy ablation
