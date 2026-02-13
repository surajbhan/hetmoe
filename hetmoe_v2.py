"""
Heterogeneous MoE v2 — Fixed Experiment
=========================================
Key fixes from v1:
  1. NO random projections — experts see raw structural data
  2. All data natively 1024-dim (32×32 grids, 1024-step sequences)
  3. Fixed big_homo bug (now actually scales up)
  4. Added param-matched hetero (scales UP hetero to match homo)
  5. Scale test across 3 expert sizes

Hardware: 4GB GPU (GTX 1650 tested)

Usage:
    python hetmoe_v2.py              # Full run
    python hetmoe_v2.py --quick      # Quick validation (~3 min)
    python hetmoe_v2.py --cpu        # Force CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time
import json
from dataclasses import dataclass, field

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Data — all types produce 1024-dim vectors with preserved structure
    grid_side: int = 32              # 32×32 = 1024
    seq_len: int = 1024              # 1024-step time series
    freq_bins: int = 1024            # 1024 frequency bins
    input_dim: int = 1024            # = grid_side² = seq_len = freq_bins

    # Model
    expert_hidden: int = 128         # base expert hidden dim
    router_hidden: int = 64
    n_experts: int = 4

    # Training
    batch_size: int = 64             # smaller batch for 4GB GPU with 1024-dim
    epochs: int = 400
    lr: float = 1e-3
    eval_samples: int = 2000

    # Balancing (weak — proven best in v1)
    bal_weight: float = 0.05
    ent_weight: float = 0.02

    quick: bool = False
    device: str = "auto"


def get_config():
    cfg = Config()
    if "--quick" in sys.argv:
        cfg.quick = True
        cfg.epochs = 100
        cfg.eval_samples = 800
        cfg.batch_size = 48
    if "--cpu" in sys.argv:
        cfg.device = "cpu"
    return cfg


# ════════════════════════════════════════════════════════════
#  DATA GENERATORS — Raw structure preserved, NO projection
# ════════════════════════════════════════════════════════════

TYPE_NAMES = ["Spatial", "Temporal", "Spectral", "Relational"]


class DataGenerator:
    """
    4 data types, each 1024-dim, with structure preserved:

    1. Spatial (32×32 grids): Multi-step heat diffusion with sources/sinks.
       Experts that reshape to 2D and convolve should excel.

    2. Temporal (1024-step): Multi-frequency signals with amplitude modulation,
       phase shifts, and autoregressive dynamics. Sequential processing helps.

    3. Spectral (1024 freq bins): Signals with sharp frequency peaks buried
       in noise. FFT-based processing gives direct advantage.

    4. Relational (32×32 distance matrix): Clustered point clouds with
       varying cluster count and spread. Global structure matters.

    All data is flattened to 1024-dim. The STRUCTURE is in the ordering
    of elements — spatial data has 2D locality, temporal has sequential
    locality, spectral has frequency-domain peaks, relational has
    block-diagonal structure.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        # Pre-compute diffusion kernel
        self.diff_kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0

    @torch.no_grad()
    def gen_spatial(self, n):
        """32×32 heat diffusion with point sources → predict center region mean."""
        s = self.cfg.grid_side
        grids = torch.randn(n, 1, s, s, device=self.device) * 0.3

        # Add 2-5 point sources/sinks at random locations
        for i in range(n):
            n_sources = torch.randint(2, 6, (1,)).item()
            for _ in range(n_sources):
                cx = torch.randint(2, s - 2, (1,)).item()
                cy = torch.randint(2, s - 2, (1,)).item()
                strength = (torch.rand(1, device=self.device).item() - 0.5) * 4.0
                # Gaussian blob
                yy, xx = torch.meshgrid(torch.arange(s, device=self.device),
                                         torch.arange(s, device=self.device), indexing='ij')
                blob = strength * torch.exp(-((xx - cx)**2 + (yy - cy)**2).float() / 8.0)
                grids[i, 0] += blob

        # Multi-step diffusion (5 steps of 3×3 averaging)
        for _ in range(5):
            grids = F.conv2d(F.pad(grids, (1, 1, 1, 1), mode='circular'), self.diff_kernel)

        # Target: mean of center 8×8 region after ONE MORE diffusion step
        final = F.conv2d(F.pad(grids, (1, 1, 1, 1), mode='circular'), self.diff_kernel)
        c = s // 2
        targets = final[:, 0, c-4:c+4, c-4:c+4].mean(dim=(1, 2))

        inputs = grids.squeeze(1).reshape(n, -1)  # (n, 1024) — raw grid, structure preserved
        return inputs, targets

    @torch.no_grad()
    def gen_temporal(self, n):
        """1024-step multi-frequency signal → predict weighted future average."""
        T = self.cfg.seq_len
        t = torch.arange(T, dtype=torch.float32, device=self.device).unsqueeze(0) / T

        signals = torch.zeros(n, T, device=self.device)

        # 3-5 sinusoidal components with different frequencies
        n_components = torch.randint(3, 6, (n,))
        for i in range(n):
            for _ in range(n_components[i].item()):
                freq = torch.rand(1, device=self.device).item() * 20 + 1  # 1-21 Hz
                amp = torch.rand(1, device=self.device).item() * 0.5 + 0.2
                phase = torch.rand(1, device=self.device).item() * 2 * np.pi
                signals[i] += amp * torch.sin(2 * np.pi * freq * t[0] + phase)

        # Amplitude modulation (slow envelope)
        env_freq = torch.rand(n, 1, device=self.device) * 2 + 0.5
        envelope = 0.5 + 0.5 * torch.sin(2 * np.pi * env_freq * t)
        signals = signals * envelope

        # AR(5) component
        ar_coeffs = torch.rand(n, 5, device=self.device) * 0.15
        for step in range(5, T):
            ar = (signals[:, step-5:step] * ar_coeffs).sum(dim=1)
            signals[:, step] = signals[:, step] + ar

        # Noise
        signals = signals + torch.randn(n, T, device=self.device) * 0.05

        # Target: exponentially-weighted mean of last 64 steps
        weights = torch.exp(torch.linspace(-3, 0, 64, device=self.device))
        weights = weights / weights.sum()
        targets = (signals[:, -64:] * weights.unsqueeze(0)).sum(dim=1)

        return signals, targets  # (n, 1024) — raw sequence, structure preserved

    @torch.no_grad()
    def gen_spectral(self, n):
        """Signal with sharp spectral peaks → predict dominant frequency."""
        F_ = self.cfg.freq_bins

        spectra = torch.zeros(n, F_, device=self.device)
        dominant_freqs = torch.zeros(n, device=self.device)

        for i in range(n):
            # Background pink noise (1/f)
            freqs_idx = torch.arange(1, F_ + 1, dtype=torch.float32, device=self.device)
            spectra[i] = 0.1 / torch.sqrt(freqs_idx) * torch.randn(F_, device=self.device).abs()

            # Add 2-4 sharp peaks at random frequencies
            n_peaks = torch.randint(2, 5, (1,)).item()
            max_amp = 0
            for _ in range(n_peaks):
                center = torch.randint(10, F_ - 10, (1,)).item()
                width = torch.randint(2, 6, (1,)).item()
                amplitude = torch.rand(1, device=self.device).item() * 3.0 + 1.0

                # Gaussian peak
                bins = torch.arange(F_, dtype=torch.float32, device=self.device)
                peak = amplitude * torch.exp(-((bins - center) ** 2) / (2 * width ** 2))
                spectra[i] += peak

                if amplitude > max_amp:
                    max_amp = amplitude
                    dominant_freqs[i] = center / F_  # normalize to [0, 1]

        # Add measurement noise
        spectra = spectra + torch.randn(n, F_, device=self.device).abs() * 0.05

        return spectra, dominant_freqs  # (n, 1024) — raw spectrum, structure preserved

    @torch.no_grad()
    def gen_relational(self, n):
        """Clustered point clouds → predict mean inter-cluster distance."""
        s = self.cfg.grid_side  # 32 points
        points = torch.zeros(n, s, 2, device=self.device)
        targets = torch.zeros(n, device=self.device)

        for i in range(n):
            n_clusters = torch.randint(2, 6, (1,)).item()
            centers = torch.rand(n_clusters, 2, device=self.device) * 4 - 2
            spread = torch.rand(1, device=self.device).item() * 0.4 + 0.05
            assignments = torch.randint(0, n_clusters, (s,))

            for j in range(s):
                points[i, j] = centers[assignments[j]] + torch.randn(2, device=self.device) * spread

            # Target: mean distance between cluster centers
            if n_clusters > 1:
                cdiff = centers.unsqueeze(0) - centers.unsqueeze(1)
                cdist = torch.sqrt((cdiff ** 2).sum(-1) + 1e-8)
                # Mean off-diagonal
                mask = ~torch.eye(n_clusters, dtype=torch.bool, device=self.device)
                targets[i] = cdist[mask].mean() / 4.0  # normalize
            else:
                targets[i] = 0.0

        # Build pairwise distance matrix (32×32 = 1024 when flattened)
        diff = points.unsqueeze(2) - points.unsqueeze(1)  # (n, 32, 32, 2)
        dist_matrix = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (n, 32, 32)
        inputs = dist_matrix.reshape(n, -1)  # (n, 1024)

        return inputs, targets

    def mixed_batch(self, n):
        k = n // 4
        sizes = [k, k, k, n - 3 * k]

        data, targets, labels = [], [], []
        generators = [self.gen_spatial, self.gen_temporal,
                      self.gen_spectral, self.gen_relational]

        for type_idx, (gen_fn, size) in enumerate(zip(generators, sizes)):
            x, y = gen_fn(size)
            data.append(x)
            targets.append(y)
            labels.append(torch.full((size,), type_idx, dtype=torch.long, device=self.device))

        data = torch.cat(data)
        targets = torch.cat(targets)
        labels = torch.cat(labels)

        perm = torch.randperm(len(data), device=self.device)
        return data[perm], targets[perm], labels[perm]


# ════════════════════════════════════════════════════════════
#  EXPERT ARCHITECTURES — Now working on RAW structured data
# ════════════════════════════════════════════════════════════

class FFNExpert(nn.Module):
    """Standard MLP. No structural assumptions."""
    name_tag = "FFN"

    def __init__(self, din, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid),
            nn.GELU(),
            nn.Linear(hid, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SpatialExpert(nn.Module):
    """
    Reshapes 1024-dim input to 32×32 grid → 2D convolutions.
    Inductive bias: local 2D spatial correlations.
    """
    name_tag = "Spatial2D"

    def __init__(self, din, hid):
        super().__init__()
        self.grid_side = int(np.sqrt(din))
        ch = max(hid // 8, 8)

        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 5, padding=2),        # large kernel for diffusion
            nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),                # → (ch*2, 4, 4)
        )
        self.head = nn.Sequential(
            nn.Linear(ch * 2 * 16, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        s = self.grid_side
        grid = x[:, :s * s].reshape(B, 1, s, s)
        h = self.conv(grid).reshape(B, -1)
        return self.head(h).squeeze(-1)


class TemporalExpert(nn.Module):
    """
    Treats 1024-dim input as a sequence → dilated causal convolutions.
    Inductive bias: multi-scale temporal patterns.
    """
    name_tag = "TempConv"

    def __init__(self, din, hid):
        super().__init__()
        ch = max(hid // 8, 8)

        # Dilated conv stack (receptive field: 1 + 2 + 4 + 8 + 16 = 31 per side)
        self.convs = nn.ModuleList([
            nn.Conv1d(1, ch, 7, padding=3, dilation=1),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4),
            nn.Conv1d(ch, ch, 3, padding=8, dilation=8),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(ch) for _ in range(4)])

        self.head = nn.Sequential(
            nn.Linear(ch, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        h = x.unsqueeze(1)  # (B, 1, 1024)

        for conv, norm in zip(self.convs, self.norms):
            h = F.gelu(norm(conv(h)))

        h = h.mean(dim=-1)  # global avg pool → (B, ch)
        return self.head(h).squeeze(-1)


class SpectralExpert(nn.Module):
    """
    Applies FFT then processes magnitude spectrum.
    Inductive bias: frequency-domain analysis.
    """
    name_tag = "Spectral"

    def __init__(self, din, hid):
        super().__init__()
        freq_dim = din // 2 + 1  # rfft output size

        # Process in frequency domain
        self.freq_conv = nn.Sequential(
            nn.Conv1d(1, hid // 4, 7, padding=3),   # detect spectral peaks
            nn.GELU(),
            nn.Conv1d(hid // 4, hid // 4, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),                # compress to 16 freq bands
        )
        self.head = nn.Sequential(
            nn.Linear(hid // 4 * 16, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        # Apply FFT
        fft = torch.fft.rfft(x, dim=-1)
        magnitude = torch.abs(fft).unsqueeze(1)  # (B, 1, freq_dim)

        h = self.freq_conv(magnitude).reshape(B, -1)
        return self.head(h).squeeze(-1)


class RelationalExpert(nn.Module):
    """
    Reshapes to 32×32 → self-attention over 32 tokens of dim 32.
    Inductive bias: pairwise relationships, global structure.
    """
    name_tag = "Relational"

    def __init__(self, din, hid):
        super().__init__()
        self.n_tokens = int(np.sqrt(din))
        self.token_dim = self.n_tokens

        attn_dim = max(hid // 4, 16)
        self.proj = nn.Linear(self.token_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.head = nn.Sequential(
            nn.Linear(attn_dim, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = x.reshape(B, self.n_tokens, self.token_dim)
        h = self.proj(tokens)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)   # residual
        h = h.mean(dim=1)             # pool over tokens
        return self.head(h).squeeze(-1)


# ════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════

class Router(nn.Module):
    def __init__(self, din, n_experts, hidden=64):
        super().__init__()
        # Router sees compressed input (avoid 1024→hidden being too expensive)
        self.compress = nn.Sequential(
            nn.Linear(din, hidden * 2),
            nn.GELU(),
        )
        self.gate = nn.Linear(hidden * 2, n_experts)

    def forward(self, x):
        h = self.compress(x)
        logits = self.gate(h)
        weights = F.softmax(logits, dim=-1)
        return weights


# ════════════════════════════════════════════════════════════
#  MOE MODEL
# ════════════════════════════════════════════════════════════

class HetMoE(nn.Module):
    def __init__(self, experts, names, din, cfg):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.names = list(names)
        self.router = Router(din, len(experts), cfg.router_hidden)
        self.bal_weight = cfg.bal_weight
        self.ent_weight = cfg.ent_weight

    def forward(self, x):
        weights = self.router(x)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        output = (weights * expert_outs).sum(dim=1)

        avg_w = weights.mean(dim=0)
        uniform = torch.ones_like(avg_w) / len(self.experts)
        bal_loss = self.bal_weight * ((avg_w - uniform) ** 2).sum()
        ent_loss = -self.ent_weight * (avg_w * torch.log(avg_w + 1e-8)).sum()

        return output, weights, bal_loss + ent_loss

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════

def train_model(model, datagen, cfg, name=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    losses = []
    type_losses = [[] for _ in range(4)]
    log_every = max(cfg.epochs // 6, 1)

    for ep in range(cfg.epochs):
        model.train()
        x, y, lab = datagen.mixed_batch(cfg.batch_size)

        pred, weights, aux_loss = model(x)
        task_loss = F.mse_loss(pred, y)
        loss = task_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(task_loss.item())
        with torch.no_grad():
            for t in range(4):
                m = lab == t
                if m.any():
                    type_losses[t].append(F.mse_loss(pred[m], y[m]).item())

        if (ep + 1) % log_every == 0:
            tl = " | ".join(f"{TYPE_NAMES[t]}={type_losses[t][-1]:.4f}"
                            for t in range(4) if type_losses[t])
            print(f"  [{name}] ep {ep+1:>4d}/{cfg.epochs}  "
                  f"loss={task_loss.item():.4f}  {tl}")

    return losses, type_losses


@torch.no_grad()
def evaluate_model(model, datagen, cfg, name=""):
    model.eval()
    all_preds, all_targets, all_labels, all_weights = [], [], [], []

    remaining = cfg.eval_samples
    while remaining > 0:
        n = min(remaining, cfg.batch_size)
        x, y, lab = datagen.mixed_batch(n)
        pred, weights, _ = model(x)
        all_preds.append(pred)
        all_targets.append(y)
        all_labels.append(lab)
        all_weights.append(weights)
        remaining -= n

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    labels = torch.cat(all_labels)
    weights = torch.cat(all_weights)

    overall = F.mse_loss(preds, targets).item()
    per_type = {}
    routing = torch.zeros(4, len(model.names))

    for t in range(4):
        mask = labels == t
        if mask.any():
            per_type[TYPE_NAMES[t]] = F.mse_loss(preds[mask], targets[mask]).item()
            routing[t] = weights[mask].mean(dim=0).cpu()

    print(f"\n  [{name}]  Overall: {overall:.4f}  |  Params: {model.get_param_count():,}")
    for t in TYPE_NAMES:
        print(f"    {t:<12s}: {per_type.get(t, -1):.4f}")
    print(f"  Routing weights:")
    print(f"    {'':12s}" + "".join(f"{n:<13s}" for n in model.names))
    for t in range(4):
        row = f"    {TYPE_NAMES[t]:<12s}" + "".join(f"{routing[t, j].item():<13.3f}"
                                                     for j in range(routing.shape[1]))
        print(row)

    return {
        "overall": overall,
        "per_type": per_type,
        "routing": routing.numpy().tolist(),
        "params": model.get_param_count()
    }


# ════════════════════════════════════════════════════════════
#  BUILD MODELS
# ════════════════════════════════════════════════════════════

def build_homo(cfg, hid=None):
    h = hid or cfg.expert_hidden
    return HetMoE(
        [FFNExpert(cfg.input_dim, h) for _ in range(4)],
        [f"FFN_{i}" for i in range(4)],
        cfg.input_dim, cfg
    )


def build_hetero(cfg, hid=None):
    h = hid or cfg.expert_hidden
    return HetMoE(
        [SpatialExpert(cfg.input_dim, h),
         TemporalExpert(cfg.input_dim, h),
         SpectralExpert(cfg.input_dim, h),
         RelationalExpert(cfg.input_dim, h)],
        ["Spatial", "Temporal", "Spectral", "Relational"],
        cfg.input_dim, cfg
    )


def build_hybrid(cfg, hid=None):
    h = hid or cfg.expert_hidden
    return HetMoE(
        [FFNExpert(cfg.input_dim, h),
         SpatialExpert(cfg.input_dim, h),
         TemporalExpert(cfg.input_dim, h),
         SpectralExpert(cfg.input_dim, h)],
        ["FFN", "Spatial", "Temporal", "Spectral"],
        cfg.input_dim, cfg
    )


# ════════════════════════════════════════════════════════════
#  EXPERIMENTS
# ════════════════════════════════════════════════════════════

def experiment_main(cfg, device, datagen):
    """Core comparison: Homo vs Hetero vs Hybrid, controlled."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: Core Comparison (same hidden dim)")
    print("=" * 65)

    results = {}

    for build_fn, key, label in [
        (build_homo, "homo", "Homo 4×FFN"),
        (build_hetero, "hetero", "Hetero (Spat+Temp+Spec+Rel)"),
        (build_hybrid, "hybrid", "Hybrid (FFN+Spat+Temp+Spec)"),
    ]:
        print(f"\n  ── {label} ──")
        torch.manual_seed(42)
        np.random.seed(42)
        model = build_fn(cfg).to(device)
        losses, tl = train_model(model, datagen, cfg, label)
        results[key] = evaluate_model(model, datagen, cfg, label)
        results[key]["losses"] = losses
        results[key]["type_losses"] = tl
        del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    return results


def experiment_param_match(cfg, device, datagen, homo_params, hetero_params):
    """Fix the big_homo bug + add param-matched hetero."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: Parameter-Matched Comparisons")
    print("=" * 65)

    results = {}

    # Big Homo: scale UP hidden dim to 256
    big_hid = 256
    print(f"\n  ── Big Homo (hidden={big_hid}) ──")
    torch.manual_seed(42)
    np.random.seed(42)
    model = build_homo(cfg, hid=big_hid).to(device)
    losses, tl = train_model(model, datagen, cfg, f"Big Homo (h={big_hid})")
    results["big_homo"] = evaluate_model(model, datagen, cfg, f"Big Homo (h={big_hid})")
    results["big_homo"]["losses"] = losses
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    # Param-matched Hetero: scale UP hetero hidden to match homo params
    # homo at hid=128 has X params, hetero at hid=128 has Y params
    # scale hetero hidden so its params ≈ homo params
    ratio = homo_params / max(hetero_params, 1)
    scaled_hid = int(cfg.expert_hidden * np.sqrt(ratio))  # sqrt because params ~ hid²
    scaled_hid = min(scaled_hid, 320)  # cap for GPU memory

    print(f"\n  ── Param-Matched Hetero (hidden={scaled_hid}) ──")
    torch.manual_seed(42)
    np.random.seed(42)
    model = build_hetero(cfg, hid=scaled_hid).to(device)
    losses, tl = train_model(model, datagen, cfg, f"Hetero (h={scaled_hid})")
    results["hetero_matched"] = evaluate_model(model, datagen, cfg, f"Hetero (h={scaled_hid})")
    results["hetero_matched"]["losses"] = losses
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    # Param-matched Hybrid
    print(f"\n  ── Param-Matched Hybrid (hidden={scaled_hid}) ──")
    torch.manual_seed(42)
    np.random.seed(42)
    model = build_hybrid(cfg, hid=scaled_hid).to(device)
    losses, tl = train_model(model, datagen, cfg, f"Hybrid (h={scaled_hid})")
    results["hybrid_matched"] = evaluate_model(model, datagen, cfg, f"Hybrid (h={scaled_hid})")
    results["hybrid_matched"]["losses"] = losses
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    return results


def experiment_seeds(cfg, device, datagen):
    """Robustness across 5 seeds."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: Robustness (5 seeds)")
    print("=" * 65)

    seeds = [42, 123, 456, 789, 1337]
    results = {"homo": [], "hetero": [], "hybrid": []}

    for seed in seeds:
        for build_fn, key in [(build_homo, "homo"), (build_hetero, "hetero"), (build_hybrid, "hybrid")]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_fn(cfg).to(device)
            train_model(model, datagen, cfg, f"{key} s={seed}")
            res = evaluate_model(model, datagen, cfg, f"{key} s={seed}")
            results[key].append(res["overall"])
            del model; torch.cuda.empty_cache() if device.type == "cuda" else None

        print(f"  Seed {seed}: H={results['homo'][-1]:.4f}  "
              f"Het={results['hetero'][-1]:.4f}  "
              f"Hyb={results['hybrid'][-1]:.4f}")

    results["seeds"] = seeds
    return results


def experiment_scale(cfg, device, datagen):
    """Does hetero advantage grow with scale?"""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4: Scale Test (small → medium → large)")
    print("=" * 65)

    hidden_dims = [64, 128, 256]
    results = {"dims": hidden_dims, "homo": [], "hetero": [], "hybrid": []}

    for hid in hidden_dims:
        print(f"\n  ── Hidden dim = {hid} ──")
        for build_fn, key in [(build_homo, "homo"), (build_hetero, "hetero"), (build_hybrid, "hybrid")]:
            torch.manual_seed(42)
            np.random.seed(42)
            model = build_fn(cfg, hid=hid).to(device)
            train_model(model, datagen, cfg, f"{key} h={hid}")
            res = evaluate_model(model, datagen, cfg, f"{key} h={hid}")
            results[key].append({"hid": hid, "mse": res["overall"], "params": res["params"],
                                 "per_type": res["per_type"]})
            del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    return results


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

def make_plots(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    exp1 = all_results["exp1"]
    COLORS = {"homo": "#3498db", "hetero": "#e67e22", "hybrid": "#2ecc71",
              "big_homo": "#95a5a6", "hetero_matched": "#9b59b6", "hybrid_matched": "#e74c3c"}

    # Fig 1: Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for key, label in [("homo", "Homo"), ("hetero", "Hetero"), ("hybrid", "Hybrid")]:
        L = exp1[key]["losses"]
        # Smooth
        w = min(20, len(L) // 5)
        if w > 1:
            smoothed = np.convolve(L, np.ones(w)/w, mode='valid')
        else:
            smoothed = L
        ax1.plot(smoothed, label=label, color=COLORS[key], alpha=0.8)
    ax1.set(xlabel="Epoch", ylabel="MSE", title="(a) Training Loss")
    ax1.set_yscale("log"); ax1.legend(); ax1.grid(alpha=0.3)

    # Per-type for hetero
    for t in range(4):
        tl = exp1["hetero"]["type_losses"][t]
        if tl:
            w = min(20, len(tl) // 5)
            smoothed = np.convolve(tl, np.ones(w)/w, mode='valid') if w > 1 else tl
            ax2.plot(smoothed, label=TYPE_NAMES[t], alpha=0.7)
    ax2.set(xlabel="Epoch", ylabel="MSE", title="(b) Per-Type (Hetero)")
    ax2.set_yscale("log"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_training.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: Routing heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (key, title) in enumerate([("homo", "Homo 4×FFN"),
                                         ("hetero", "Hetero"),
                                         ("hybrid", "Hybrid")]):
        ax = axes[idx]
        rt = np.array(exp1[key]["routing"])
        names = exp1[key].get("names_list", None)
        if key == "homo":
            names = ["FFN_0", "FFN_1", "FFN_2", "FFN_3"]
        elif key == "hetero":
            names = ["Spatial", "Temporal", "Spectral", "Relational"]
        else:
            names = ["FFN", "Spatial", "Temporal", "Spectral"]

        im = ax.imshow(rt, cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(TYPE_NAMES, fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(4):
            for j in range(rt.shape[1]):
                c = "white" if rt[i, j] > 0.35 else "black"
                ax.text(j, i, f"{rt[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=c)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_routing.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: Per-type bar comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    models = {"Homo": exp1["homo"], "Hetero": exp1["hetero"], "Hybrid": exp1["hybrid"]}
    cols = [COLORS["homo"], COLORS["hetero"], COLORS["hybrid"]]
    for idx, dtype in enumerate(TYPE_NAMES):
        ax = axes[idx]
        mses = [m["per_type"].get(dtype, 0) for m in models.values()]
        bars = ax.bar(range(len(models)), mses, color=cols, edgecolor="black", linewidth=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
                    f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(list(models.keys()), fontsize=9)
        ax.set(ylabel="MSE", title=dtype)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-Type Performance (v2 — raw structure preserved)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_per_type.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: Scale test
    if "exp4" in all_results:
        exp4 = all_results["exp4"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for key, label in [("homo", "Homo"), ("hetero", "Hetero"), ("hybrid", "Hybrid")]:
            params = [r["params"] for r in exp4[key]]
            mses = [r["mse"] for r in exp4[key]]
            ax1.plot(params, mses, 'o-', label=label, color=COLORS[key], markersize=8)
        ax1.set(xlabel="Parameters", ylabel="MSE", title="(a) MSE vs Parameters")
        ax1.legend(); ax1.grid(alpha=0.3)

        hids = exp4["dims"]
        for key, label in [("homo", "Homo"), ("hetero", "Hetero"), ("hybrid", "Hybrid")]:
            mses = [r["mse"] for r in exp4[key]]
            ax2.plot(hids, mses, 'o-', label=label, color=COLORS[key], markersize=8)
        ax2.set(xlabel="Expert Hidden Dim", ylabel="MSE", title="(b) MSE vs Expert Size")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig4_scaling.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  All figures saved to {out_dir}/")


# ════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════

def print_final_summary(all_results):
    print("\n" + "=" * 70)
    print("  ╔══════════════════════════════════════════╗")
    print("  ║        FINAL RESULTS SUMMARY (v2)        ║")
    print("  ╚══════════════════════════════════════════╝")
    print("=" * 70)

    exp1 = all_results["exp1"]

    print(f"\n  ┌─ EXPERIMENT 1: Core Comparison ─────────────────────────")
    print(f"  │ {'Model':<25s} {'Params':>8s} {'Overall':>8s}", end="")
    for t in TYPE_NAMES:
        print(f" {t[:7]:>8s}", end="")
    print()
    print(f"  │ " + "─" * 63)

    for key, name in [("homo", "Homo 4×FFN"), ("hetero", "Hetero"), ("hybrid", "Hybrid")]:
        r = exp1[key]
        print(f"  │ {name:<25s} {r['params']:>8,} {r['overall']:>8.4f}", end="")
        for t in TYPE_NAMES:
            print(f" {r['per_type'].get(t, 0):>8.4f}", end="")
        print()

    baseline = exp1["homo"]["overall"]
    for key, name in [("hetero", "Hetero"), ("hybrid", "Hybrid")]:
        imp = (baseline - exp1[key]["overall"]) / baseline * 100
        print(f"  │ {name} vs Homo: {imp:+.1f}%")
    print(f"  └──────────────────────────────────────────────────────")

    if "exp2" in all_results:
        exp2 = all_results["exp2"]
        print(f"\n  ┌─ EXPERIMENT 2: Param-Matched ─────────────────────────")
        for key, name in [("big_homo", "Big Homo"), ("hetero_matched", "Hetero (matched)"),
                           ("hybrid_matched", "Hybrid (matched)")]:
            if key in exp2:
                r = exp2[key]
                print(f"  │ {name:<25s} {r['params']:>8,} {r['overall']:>8.4f}")
        print(f"  └──────────────────────────────────────────────────────")

    if "exp3" in all_results:
        exp3 = all_results["exp3"]
        print(f"\n  ┌─ EXPERIMENT 3: Robustness ────────────────────────────")
        for key in ["homo", "hetero", "hybrid"]:
            vals = exp3[key]
            print(f"  │ {key:<10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        het_wins = sum(1 for h, t in zip(exp3["homo"], exp3["hetero"]) if t < h)
        hyb_wins = sum(1 for h, t in zip(exp3["homo"], exp3["hybrid"]) if t < h)
        print(f"  │ Hetero wins: {het_wins}/{len(exp3['homo'])}  |  Hybrid wins: {hyb_wins}/{len(exp3['homo'])}")
        print(f"  └──────────────────────────────────────────────────────")

    if "exp4" in all_results:
        exp4 = all_results["exp4"]
        print(f"\n  ┌─ EXPERIMENT 4: Scaling ──────────────────────────────")
        for hid_idx, hid in enumerate(exp4["dims"]):
            print(f"  │ Hidden={hid}:")
            for key in ["homo", "hetero", "hybrid"]:
                r = exp4[key][hid_idx]
                print(f"  │   {key:<10s}: MSE={r['mse']:.4f}  params={r['params']:,}")
        print(f"  └──────────────────────────────────────────────────────")

    print("=" * 70)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    cfg = get_config()

    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {torch.cuda.get_device_name()} ({gpu_mem:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("  Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("  Using CPU")
    else:
        device = torch.device(cfg.device)

    print(f"\n{'='*65}")
    print(f"  HETEROGENEOUS MoE v2 — RAW STRUCTURE EXPERIMENT")
    print(f"  Mode: {'QUICK' if cfg.quick else 'FULL'}")
    print(f"  Epochs: {cfg.epochs}  |  Input: {cfg.input_dim}  |  Batch: {cfg.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*65}")

    t0 = time.time()
    datagen = DataGenerator(cfg, device)
    all_results = {}

    # Exp 1: Core comparison
    all_results["exp1"] = experiment_main(cfg, device, datagen)

    # Exp 2: Param-matched
    homo_p = all_results["exp1"]["homo"]["params"]
    hetero_p = all_results["exp1"]["hetero"]["params"]
    all_results["exp2"] = experiment_param_match(cfg, device, datagen, homo_p, hetero_p)

    # Exp 3: Robustness
    all_results["exp3"] = experiment_seeds(cfg, device, datagen)

    # Exp 4: Scale test
    all_results["exp4"] = experiment_scale(cfg, device, datagen)

    # Results & plots
    out_dir = "hetmoe_v2_results"
    os.makedirs(out_dir, exist_ok=True)
    make_plots(all_results, out_dir)
    print_final_summary(all_results)

    # Save JSON
    save_data = {}
    for exp_key, exp_val in all_results.items():
        if isinstance(exp_val, dict):
            save_data[exp_key] = {}
            for k, v in exp_val.items():
                if isinstance(v, dict):
                    save_data[exp_key][k] = {
                        kk: vv for kk, vv in v.items()
                        if kk not in ("losses", "type_losses")
                    }
                elif isinstance(v, list):
                    save_data[exp_key][k] = v
                else:
                    save_data[exp_key][k] = v

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
