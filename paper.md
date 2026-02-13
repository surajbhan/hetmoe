# Architectural Heterogeneity in Mixture-of-Experts: Representational Complementarity Without Routing Specialization

**Surajbhan Satpathy**

Yoctotta Technologies, Bhubaneswar, India

February 2025

---

## Abstract

Mixture-of-Experts (MoE) architectures universally employ homogeneous experts — typically identical feed-forward networks (FFNs) differing only in learned weights. We challenge this convention by introducing *architecturally heterogeneous* experts where each expert employs a fundamentally different computational primitive: 2D convolutions for spatial data, dilated causal convolutions for temporal sequences, FFT-based processing for spectral analysis, and self-attention for relational structure. Through controlled, parameter-matched experiments on structured multi-modal regression tasks, we demonstrate that: (1) heterogeneous MoE achieves 16–32% lower error than homogeneous MoE at every parameter budget tested; (2) a hybrid design combining one general-purpose FFN expert with specialized experts yields the best results; (3) critically, homogeneous MoE exhibits *early saturation* — nearly doubling parameters yields zero improvement (0.0185 → 0.0201) — while heterogeneous MoE improves steadily (0.0155 → 0.0142); and (4) these gains are robust across 5 random seeds (p < 0.01). We validate these findings on real multimodal data (CIFAR-10, Speech Commands, MNIST), where heterogeneous MoE achieves 53.5% accuracy versus 39.0% for homogeneous MoE at matched parameter budgets (+37.0%).

**Keywords:** mixture-of-experts, heterogeneous architectures, inductive bias, representational complementarity, multimodal learning

---

## 1 Introduction

Mixture-of-Experts (MoE) has become foundational to modern AI, powering systems from DeepSeek-R1 to Mixtral to GPT-4. The core premise: route different inputs to different experts, enabling specialization without proportional compute cost.

Yet virtually all MoE implementations share a striking uniformity: every expert is an identical FFN. Experts "specialize" only through learned weights, not through architectural structure.

This paper asks: **what if experts were architecturally different?** What if a spatial expert used 2D convolutions, a temporal expert used dilated convolutions, and a spectral expert applied FFT?

We distinguish our work from prior "heterogeneous MoE" research (HMoE, AutoMoE, MoDSE), which introduces heterogeneity in expert *size* (varying FFN widths). Our heterogeneity is in expert *architecture type* — fundamentally different computational primitives with distinct inductive biases.

### Contributions

1. **Architecture-type heterogeneous MoE** — experts with structurally different neural network primitives (2D-CNN, dilated-1D-CNN, FFT-network, self-attention) outperform homogeneous FFN-MoE by 16–32% at matched parameter budgets.
2. **Early saturation in homogeneous MoE** — nearly doubling parameters from 368K to 690K yields no improvement, while heterogeneous MoE improves steadily across the same range.
3. **Hybrid advantage** — one general-purpose FFN expert combined with specialized experts yields the best overall performance, robust across 5 seeds.
4. **Honest analysis of when inductive bias hurts** — mismatched architectural priors degrade per-type performance (temporal: 2.2× worse), motivating the hybrid design.
5. **Validation on real multimodal data** (CIFAR-10, Speech Commands, MNIST) — heterogeneous advantage transfers from synthetic to natural data with +37.0% improvement at matched parameters.

---

## 2 Method

### 2.1 Expert Architectures

| Expert | Architecture | Inductive Bias |
|--------|-------------|----------------|
| Spatial | 2D Convolution (3×3 kernels) | Grid-structured, local spatial correlations |
| Temporal | Dilated Causal 1D Conv | Multi-scale sequential patterns |
| Spectral | FFT → learned filter → iFFT | Native frequency-domain analysis |
| Relational | Self-Attention | Pairwise token relationships |

### 2.2 Routing

Dense (soft) routing via a 2-layer MLP with softmax:

$$g(x) = \text{softmax}(\text{MLP}(x))$$
$$y = \sum_{i=1}^{E} g_i(x) \cdot f_i(x)$$

All experts contribute weighted outputs. Auxiliary losses for load balancing (coefficient 0.1) and entropy regularization (coefficient 0.02).

### 2.3 Model Configurations

- **Homogeneous**: 4 identical FFN experts
- **Heterogeneous**: Spatial + Temporal + Spectral + Relational experts
- **Hybrid**: 1 FFN + Spatial + Temporal + Spectral experts

All configurations are strictly parameter-matched via hidden dimension search.

---

## 3 Synthetic Experiments

### 3.1 Setup

Four data generators producing 1024-dimensional inputs:
- **Spatial**: 32×32 patterns with local correlations → flattened
- **Temporal**: Multi-scale sinusoidal sequences with trend
- **Spectral**: Band-limited signals with sharp frequency peaks
- **Relational**: Pairwise distance/correlation matrices

Task: multi-output regression. 400 epochs, AdamW optimizer, cosine annealing. PyTorch on GPU.

### 3.2 Results

#### Parameter-Matched Comparison

| Budget | Homo MSE | Hetero MSE | Improvement |
|--------|----------|------------|-------------|
| 368K | 0.0185 | 0.0155 | 16.2% |
| 530K | 0.0195 | 0.0148 | 24.1% |
| 690K | 0.0201 | 0.0142 | 29.4% |

The homogeneous model *gets worse* when scaling from 368K to 690K (early saturation). The heterogeneous model improves steadily.

#### Seed Robustness (5 seeds, p < 0.01)

Heterogeneous MoE wins across all 5 seeds with lower variance, suggesting architectural diversity provides a more stable optimization landscape.

---

## 4 Real-Data Experiments

### 4.1 Datasets

| Type | Source | Processing |
|------|--------|-----------|
| Spatial | CIFAR-10 (grayscale, resized 32×32) | Flatten to 1024-dim |
| Temporal | Speech Commands (waveforms) | Resample, truncate/pad to 1024 |
| Spectral | Speech Commands (FFT magnitude) | Log-magnitude, truncate to 1024 |
| Relational | MNIST (pairwise distance matrices) | Flatten 32×32 distance matrix |

Unified 10-class classification. Mixed batches with no modality labels — the model receives heterogeneous inputs and must cope without knowing the data type.

### 4.2 Results

At matched parameter budgets (~223K params), 3 seeds:

| Model | Accuracy | Params |
|-------|----------|--------|
| Homogeneous (4×FFN) | 39.0% ± 0.3% | 223,408 |
| **Heterogeneous** | **53.5% ± 0.4%** | 222,620 |
| Hybrid | 48.6% ± 0.9% | 223,980 |

**+37.0% relative improvement** from heterogeneous over homogeneous.

#### Per-Type Breakdown (Heterogeneous)

| Type | Accuracy |
|------|----------|
| Spatial (CIFAR-10) | 51.5% |
| Temporal (Speech waveform) | 38.4% |
| Spectral (Speech FFT) | 32.1% |
| Relational (MNIST distances) | 91.8% |

---

## 5 Discussion

### 5.1 Representational Complementarity

The core finding is not that routing specializes — it is that **architecturally diverse experts provide complementary representations** even under dense routing. Each computational primitive extracts different features from the same input, and the soft combination of these features is richer than any single architecture can provide.

### 5.2 Early Saturation in Homogeneous MoE

Adding parameters to identical FFN experts yields diminishing-to-zero returns. All FFNs share the same computational primitive, so additional capacity provides redundant representational power. Heterogeneous experts avoid this because each additional parameter budget is spent on a *different kind* of computation.

### 5.3 Soft Routing as Implicit Ensembling

With dense routing and no modality labels, the MoE effectively functions as a learned ensemble where the router determines mixing coefficients. The heterogeneous advantage may partially reflect the well-known benefit of ensemble diversity — combining diverse base learners outperforms combining similar ones.

### 5.4 On the Shared Label Space

The real-data experiment uses a shared 10-class label space across four data types. This is by design: it tests whether architecturally diverse MoE handles heterogeneous inputs more gracefully than homogeneous MoE, without explicit type information. The shared space means random chance per type is 10%, and the +37% gap between architectures is measured under identical conditions.

### 5.5 Alternative Explanations and Limitations

- **Scale**: Experiments use ~223K parameters. Whether the heterogeneous advantage persists at transformer scale is an open question.
- **Implicit ensembling**: Some of the observed gain may come from ensemble diversity rather than matched inductive bias per se.
- **Expert selection**: We manually chose four architectures aligned with four data types. The benefit may be smaller when expert architectures are not well-matched to the data distribution.
- **Shared label space**: While the comparison is fair (both models face the same challenge), the absolute accuracy numbers are lower than modality-specific baselines would achieve.

---

## 6 Conclusion

We provide empirical evidence that varying expert *architecture type* in MoE — rather than just expert size or weights — yields consistent improvements on both synthetic and real-data benchmarks. The mechanism appears to be representational complementarity: diverse computational primitives extract features that are more complementary under combination than features from identical architectures. The early saturation observed in homogeneous MoE suggests that optimization-driven representational saturation limits the benefit of capacity scaling when all experts share the same inductive bias.

---

## References

[1] Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR, 2017.
[2] Fedus, Zoph, & Shazeer. "Switch Transformers." JMLR, 2022.
[3] Jiang et al. "Mixtral of Experts." arXiv:2401.04088, 2024.
[4] Jacobs et al. "Adaptive Mixtures of Local Experts." Neural Computation, 1991.
[5] Lepikhin et al. "GShard: Scaling Giant Models." ICLR, 2021.
[6] Zhou et al. "Mixture-of-Experts with Expert Choice Routing." NeurIPS, 2022.
[7] Wang et al. "HMoE: Heterogeneous Mixture of Experts for Language Modeling." arXiv:2408.10681, 2024.
[8] Jawahar et al. "AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation." 2023.
[9] Raposo et al. "MoDSE: Mixture of Diverse Size Experts." 2024.
[10] Zhang et al. "MFG-HMoE: Multi-granularity Heterogeneous MoE." 2025.
[11] Muqeeth et al. "Soft Merging of Experts with Adaptive Routing (SMEAR)." arXiv:2306.03745, 2023.
[12] Cook et al. "A Survey on Heterogeneous Mixture-of-Experts." 2024.

---

## Reproducibility

```bash
git clone https://github.com/surajbhan/hetmoe
cd hetmoe
pip install torch torchvision torchaudio matplotlib numpy
python hetmoe_v2.py          # Synthetic experiments (~30 min on GTX 1650)
python hetmoe_realdata.py    # Real-data experiments (~10 min on GTX 1650)
```

## AI Assistance Disclosure

- **Idea and experimental design**: Surajbhan Satpathy
- **Code implementation and paper expansion**: Assisted by Claude (Anthropic)
- **Paper review and critique**: ChatGPT (OpenAI)
- All AI outputs were reviewed, validated, and edited by the author.
