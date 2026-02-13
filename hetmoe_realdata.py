"""
HetMoE Real-Data Experiment
============================
Tests heterogeneous MoE on REAL multimodal data:
  1. Spatial:    CIFAR-10 grayscale images (32×32=1024) — real 2D structure
  2. Temporal:   Speech Commands waveforms (resampled to 1024) — real 1D sequences
  3. Spectral:   Speech Commands FFT magnitude (1024 bins) — real frequency spectra
  4. Relational: MNIST digit similarity matrices (32×32=1024) — real pairwise structure

All modalities mapped to unified 10-class classification.
NO random projections. Data preserves native structure.

Hardware: GTX 1650 (4GB) — model is only ~200K-700K params, VRAM is trivial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import json

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
INPUT_DIM = 1024  # 32×32 for all modalities
BATCH_SIZE = 64
EPOCHS = 30  # 30 epochs with proper multi-batch training
LR = 1e-3
EVAL_SAMPLES = 4000
NUM_SEEDS = 3

# ════════════════════════════════════════════════════════════
#  DATA LOADING — Real datasets, native structure preserved
# ════════════════════════════════════════════════════════════

def load_cifar10_spatial(data_dir="./data_cache"):
    """CIFAR-10 grayscale → 32×32 = 1024-dim. Native 2D spatial structure."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.Grayscale(), T.ToTensor()])
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Flatten to 1024 but structure IS spatial (32×32 grid)
    X_train = torch.stack([img.view(-1) for img, _ in train])  # (50000, 1024)
    y_train = torch.tensor([lab for _, lab in train])
    X_test = torch.stack([img.view(-1) for img, _ in test])
    y_test = torch.tensor([lab for _, lab in test])

    # Normalize to zero mean, unit variance
    mu, std = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / (std + 1e-8)
    X_test = (X_test - mu) / (std + 1e-8)

    print(f"  CIFAR-10 spatial: train={X_train.shape}, test={X_test.shape}, classes=0-9")
    return X_train, y_train, X_test, y_test


def load_speech_commands_temporal(data_dir="./data_cache"):
    """Speech Commands waveforms → resample to 1024. Native temporal structure."""
    import torchaudio

    # 10 core commands (matches our 10-class setup)
    COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    os.makedirs(data_dir, exist_ok=True)

    train_ds = torchaudio.datasets.SPEECHCOMMANDS(root=data_dir, download=True, subset="training")
    test_ds = torchaudio.datasets.SPEECHCOMMANDS(root=data_dir, download=True, subset="testing")

    def extract(dataset, max_per_class=5000):
        X, y = [], []
        counts = {c: 0 for c in COMMANDS}
        for item in dataset:
            waveform, sr, label = item[0], item[1], item[2]
            if label not in COMMANDS:
                continue
            if counts[label] >= max_per_class:
                continue
            counts[label] += 1
            # Resample to 1024 samples — preserves temporal structure
            wav = waveform.squeeze(0)
            if len(wav) < 1024:
                wav = F.pad(wav, (0, 1024 - len(wav)))
            # Resample by interpolation to exactly 1024
            wav = F.interpolate(wav.unsqueeze(0).unsqueeze(0), size=1024, mode='linear', align_corners=False)
            wav = wav.squeeze()
            X.append(wav)
            y.append(COMMANDS.index(label))
        return torch.stack(X), torch.tensor(y)

    X_train, y_train = extract(train_ds)
    X_test, y_test = extract(test_ds)

    # Normalize
    mu, std = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / (std + 1e-8)
    X_test = (X_test - mu) / (std + 1e-8)

    print(f"  Speech temporal: train={X_train.shape}, test={X_test.shape}, classes=0-9")
    return X_train, y_train, X_test, y_test


def make_spectral_from_temporal(X_train_temp, y_train_temp, X_test_temp, y_test_temp):
    """FFT of audio waveforms → 1024-dim magnitude spectrum. Native spectral structure."""
    def to_spectrum(X):
        fft = torch.fft.rfft(X, n=2046, dim=-1)  # rfft of 2046 → 1024 bins
        mag = torch.abs(fft[:, :1024])  # take first 1024 magnitude bins
        return mag

    X_train = to_spectrum(X_train_temp)
    X_test = to_spectrum(X_test_temp)

    # Normalize
    mu, std = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / (std + 1e-8)
    X_test = (X_test - mu) / (std + 1e-8)

    print(f"  Speech spectral: train={X_train.shape}, test={X_test.shape}, classes=0-9")
    return X_train, y_train_temp.clone(), X_test, y_test_temp.clone()


def load_mnist_relational(data_dir="./data_cache"):
    """MNIST → 32×32 pairwise pixel-patch similarities = 1024-dim. Native relational structure."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.Resize(32), T.ToTensor()])
    train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    def to_relational(dataset):
        X, y = [], []
        for img, lab in dataset:
            img = img.squeeze(0)  # (32, 32)
            # Treat 32 rows as 32 "nodes" with 32-dim features
            nodes = img  # (32, 32)
            # Compute pairwise L2 distance matrix → 32×32 = 1024
            diff = nodes.unsqueeze(0) - nodes.unsqueeze(1)  # (32, 32, 32)
            dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (32, 32)
            X.append(dist.view(-1))  # (1024,)
            y.append(lab)
        return torch.stack(X), torch.tensor(y)

    X_train, y_train = to_relational(train)
    X_test, y_test = to_relational(test)

    # Normalize
    mu, std = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / (std + 1e-8)
    X_test = (X_test - mu) / (std + 1e-8)

    print(f"  MNIST relational: train={X_train.shape}, test={X_test.shape}, classes=0-9")
    return X_train, y_train, X_test, y_test


# ════════════════════════════════════════════════════════════
#  EXPERT ARCHITECTURES — Same as v2, but classification head
# ════════════════════════════════════════════════════════════

class FFNExpert(nn.Module):
    name_tag = "FFN"
    def __init__(self, din, hid, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid), nn.GELU(),
            nn.Linear(hid, hid // 2), nn.GELU(),
            nn.Linear(hid // 2, n_classes)
        )
    def forward(self, x):
        return self.net(x)


class SpatialExpert(nn.Module):
    name_tag = "Spatial2D"
    def __init__(self, din, hid, n_classes):
        super().__init__()
        self.grid_side = int(np.sqrt(din))
        ch = max(hid // 8, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 5, padding=2), nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Linear(ch * 2 * 16, hid // 2), nn.GELU(),
            nn.Linear(hid // 2, n_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        s = self.grid_side
        grid = x[:, :s * s].reshape(B, 1, s, s)
        h = self.conv(grid).reshape(B, -1)
        return self.head(h)


class TemporalExpert(nn.Module):
    name_tag = "TempConv"
    def __init__(self, din, hid, n_classes):
        super().__init__()
        ch = max(hid // 8, 8)
        self.convs = nn.ModuleList([
            nn.Conv1d(1, ch, 7, padding=3, dilation=1),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4),
            nn.Conv1d(ch, ch, 3, padding=8, dilation=8),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(ch) for _ in range(4)])
        self.head = nn.Sequential(
            nn.Linear(ch, hid // 2), nn.GELU(),
            nn.Linear(hid // 2, n_classes)
        )

    def forward(self, x):
        h = x.unsqueeze(1)
        for conv, norm in zip(self.convs, self.norms):
            h = F.gelu(norm(conv(h)))
        h = h.mean(dim=-1)
        return self.head(h)


class SpectralExpert(nn.Module):
    name_tag = "Spectral"
    def __init__(self, din, hid, n_classes):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv1d(1, hid // 4, 7, padding=3), nn.GELU(),
            nn.Conv1d(hid // 4, hid // 4, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Linear(hid // 4 * 16, hid // 2), nn.GELU(),
            nn.Linear(hid // 2, n_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        # Process raw input as spectrum (no FFT here — data is already spectral)
        h = x.unsqueeze(1)
        h = self.freq_conv(h).reshape(B, -1)
        return self.head(h)


class RelationalExpert(nn.Module):
    name_tag = "Relational"
    def __init__(self, din, hid, n_classes):
        super().__init__()
        self.n_tokens = int(np.sqrt(din))
        self.token_dim = self.n_tokens
        attn_dim = max((hid // 4 // 4) * 4, 16)  # ensure divisible by 4 heads
        self.proj = nn.Linear(self.token_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.head = nn.Sequential(
            nn.Linear(attn_dim, hid // 2), nn.GELU(),
            nn.Linear(hid // 2, n_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = x.reshape(B, self.n_tokens, self.token_dim)
        h = self.proj(tokens)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)
        h = h.mean(dim=1)
        return self.head(h)


# ════════════════════════════════════════════════════════════
#  ROUTER + MOE (classification version)
# ════════════════════════════════════════════════════════════

class Router(nn.Module):
    def __init__(self, din, n_experts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, n_experts)
        )
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class HetMoE(nn.Module):
    def __init__(self, experts, names, din, bal_w=0.05, ent_w=0.02):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.names = names
        self.router = Router(din, len(experts), 64)
        self.bal_w, self.ent_w = bal_w, ent_w

    def forward(self, x):
        w = self.router(x)  # (B, N_experts)
        # Each expert outputs (B, n_classes) logits
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, N, C)
        # Weighted combination of logits
        output = (w.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, C)

        # Auxiliary losses
        avg_w = w.mean(dim=0)
        uniform = torch.ones_like(avg_w) / len(self.experts)
        bal_loss = self.bal_w * ((avg_w - uniform) ** 2).sum()
        ent_loss = -self.ent_w * (avg_w * torch.log(avg_w + 1e-8)).sum()

        return output, w, bal_loss + ent_loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════
#  MODEL BUILDERS
# ════════════════════════════════════════════════════════════

def build_homo(hid, n_classes=NUM_CLASSES):
    return HetMoE(
        [FFNExpert(INPUT_DIM, hid, n_classes) for _ in range(4)],
        ["FFN_0", "FFN_1", "FFN_2", "FFN_3"], INPUT_DIM
    )

def build_hetero(hid, n_classes=NUM_CLASSES):
    return HetMoE(
        [SpatialExpert(INPUT_DIM, hid, n_classes),
         TemporalExpert(INPUT_DIM, hid, n_classes),
         SpectralExpert(INPUT_DIM, hid, n_classes),
         RelationalExpert(INPUT_DIM, hid, n_classes)],
        ["Spatial", "Temporal", "Spectral", "Relational"], INPUT_DIM
    )

def build_hybrid(hid, n_classes=NUM_CLASSES):
    return HetMoE(
        [FFNExpert(INPUT_DIM, hid, n_classes),
         SpatialExpert(INPUT_DIM, hid, n_classes),
         TemporalExpert(INPUT_DIM, hid, n_classes),
         SpectralExpert(INPUT_DIM, hid, n_classes)],
        ["FFN", "Spatial", "Temporal", "Spectral"], INPUT_DIM
    )


# ════════════════════════════════════════════════════════════
#  MULTIMODAL DATASET
# ════════════════════════════════════════════════════════════

class MultiModalDataset:
    """Combines 4 real datasets into mixed batches, labels unified to 0-9."""

    def __init__(self, datasets, device):
        """datasets: list of (X, y) tuples, one per modality."""
        self.device = device
        self.modalities = []
        for X, y in datasets:
            self.modalities.append((X.to(device), y.to(device)))
        self.n_mod = len(self.modalities)

    def mixed_batch(self, n):
        k = n // self.n_mod
        sizes = [k] * (self.n_mod - 1) + [n - k * (self.n_mod - 1)]

        all_x, all_y, all_types = [], [], []
        for mod_idx, ((X, y), sz) in enumerate(zip(self.modalities, sizes)):
            idx = torch.randint(0, len(X), (sz,), device=self.device)
            all_x.append(X[idx])
            all_y.append(y[idx])
            all_types.append(torch.full((sz,), mod_idx, dtype=torch.long, device=self.device))

        x = torch.cat(all_x)
        y = torch.cat(all_y)
        types = torch.cat(all_types)

        perm = torch.randperm(len(x), device=self.device)
        return x[perm], y[perm], types[perm]


# ════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════

TYPE_NAMES = ["Spatial", "Temporal", "Spectral", "Relational"]


def train_model(model, train_ds, epochs, steps_per_epoch, label=""):
    """Returns dict with 'loss' and 'acc' lists (per-epoch training curves)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    log_every = max(epochs // 4, 1)

    history = {"loss": [], "acc": []}

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0

        for step in range(steps_per_epoch):
            x, y, types = train_ds.mixed_batch(BATCH_SIZE)
            logits, weights, aux = model(x)
            loss = F.cross_entropy(logits, y) + aux

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            ep_correct += (logits.argmax(dim=-1) == y).sum().item()
            ep_total += len(y)

        avg_loss = ep_loss / steps_per_epoch
        avg_acc = ep_correct / ep_total
        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)

        if (ep + 1) % log_every == 0:
            print(f"    [{label}] ep {ep+1}/{epochs}  loss={avg_loss:.4f}  acc={avg_acc:.3f}")

    return history


@torch.no_grad()
def evaluate_model(model, test_ds, label=""):
    model.eval()
    all_correct, all_total = 0, 0
    per_type_correct = [0] * 4
    per_type_total = [0] * 4
    all_weights = []

    remaining = EVAL_SAMPLES
    while remaining > 0:
        n = min(remaining, BATCH_SIZE)
        x, y, types = test_ds.mixed_batch(n)
        logits, weights, _ = model(x)
        preds = logits.argmax(dim=-1)

        correct = (preds == y)
        all_correct += correct.sum().item()
        all_total += n
        all_weights.append(weights)

        for t in range(4):
            mask = types == t
            if mask.any():
                per_type_correct[t] += correct[mask].sum().item()
                per_type_total[t] += mask.sum().item()
        remaining -= n

    overall_acc = all_correct / all_total
    per_type_acc = {}
    for t in range(4):
        if per_type_total[t] > 0:
            per_type_acc[TYPE_NAMES[t]] = per_type_correct[t] / per_type_total[t]

    # Routing analysis
    weights = torch.cat(all_weights)
    # Re-evaluate to get types aligned (simpler: just report overall routing)

    print(f"  [{label}]  Acc={overall_acc:.4f}  Params={model.param_count():,}")
    for t in TYPE_NAMES:
        print(f"    {t:<12s}: {per_type_acc.get(t, 0):.4f}")

    return {
        "accuracy": overall_acc,
        "per_type": per_type_acc,
        "params": model.param_count(),
    }


# ════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print(f"  HETMOE REAL-DATA EXPERIMENT")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
    else:
        print(f"  WARNING: Running on CPU! Check CUDA installation.")
    print(f"{'='*65}")

    t0 = time.time()

    # ── Load real datasets ──
    print("\n  Loading datasets...")

    # 1. CIFAR-10 → spatial (32×32 grayscale images)
    cifar_Xtr, cifar_ytr, cifar_Xte, cifar_yte = load_cifar10_spatial()

    # 2. Speech Commands → temporal (1024-sample waveforms)
    speech_Xtr, speech_ytr, speech_Xte, speech_yte = load_speech_commands_temporal()

    # 3. Same audio → spectral (FFT magnitude)
    spec_Xtr, spec_ytr, spec_Xte, spec_yte = make_spectral_from_temporal(
        speech_Xtr, speech_ytr, speech_Xte, speech_yte
    )

    # 4. MNIST → relational (pairwise distance matrices)
    mnist_Xtr, mnist_ytr, mnist_Xte, mnist_yte = load_mnist_relational()

    # Balance dataset sizes (use min across modalities)
    min_train = min(len(cifar_Xtr), len(speech_Xtr), len(spec_Xtr), len(mnist_Xtr))
    min_test = min(len(cifar_Xte), len(speech_Xte), len(spec_Xte), len(mnist_Xte))
    print(f"\n  Balanced sizes: train={min_train}/modality, test={min_test}/modality")

    train_datasets = [
        (cifar_Xtr[:min_train], cifar_ytr[:min_train]),
        (speech_Xtr[:min_train], speech_ytr[:min_train]),
        (spec_Xtr[:min_train], spec_ytr[:min_train]),
        (mnist_Xtr[:min_train], mnist_ytr[:min_train]),
    ]
    test_datasets = [
        (cifar_Xte[:min_test], cifar_yte[:min_test]),
        (speech_Xte[:min_test], speech_yte[:min_test]),
        (spec_Xte[:min_test], spec_yte[:min_test]),
        (mnist_Xte[:min_test], mnist_yte[:min_test]),
    ]

    train_ds = MultiModalDataset(train_datasets, DEVICE)
    test_ds = MultiModalDataset(test_datasets, DEVICE)

    # Compute steps per epoch: one pass through all training data
    total_train = min_train * 4  # 4 modalities
    steps_per_epoch = min(total_train // BATCH_SIZE, 500)  # cap at 500 to keep runtime ~20min
    print(f"  Steps per epoch: {steps_per_epoch} ({EPOCHS} epochs = {steps_per_epoch * EPOCHS:,} total steps)")

    # ── Find param-matched hidden dims ──
    print("\n  Finding parameter-matched configurations...")
    # Hetero at h=128 is our baseline param count
    ref_model = build_hetero(128)
    ref_params = ref_model.param_count()
    print(f"  Hetero h=128: {ref_params:,} params")
    del ref_model

    # Search for homo hidden dim that gives similar params
    best_homo_hid = 20
    best_homo_diff = abs(build_homo(20).param_count() - ref_params)
    for h in range(16, 100, 2):
        p = build_homo(h).param_count()
        diff = abs(p - ref_params)
        if diff < best_homo_diff:
            best_homo_hid = h
            best_homo_diff = diff

    # Search for hybrid hidden dim
    best_hybrid_hid = 60
    best_hybrid_diff = abs(build_hybrid(60).param_count() - ref_params)
    for h in range(30, 140, 2):
        p = build_hybrid(h).param_count()
        diff = abs(p - ref_params)
        if diff < best_hybrid_diff:
            best_hybrid_hid = h
            best_hybrid_diff = diff

    print(f"  Homo h={best_homo_hid}: {build_homo(best_homo_hid).param_count():,} params")
    print(f"  Hetero h=128: {build_hetero(128).param_count():,} params")
    print(f"  Hybrid h={best_hybrid_hid}: {build_hybrid(best_hybrid_hid).param_count():,} params")

    # ── Run experiments ──
    all_results = {}

    for seed_idx, seed in enumerate([42, 123, 456][:NUM_SEEDS]):
        print(f"\n{'='*65}")
        print(f"  SEED {seed} ({seed_idx+1}/{NUM_SEEDS})")
        print(f"{'='*65}")

        seed_results = {}

        for name, build_fn, hid in [
            ("homo", build_homo, best_homo_hid),
            ("hetero", build_hetero, 128),
            ("hybrid", build_hybrid, best_hybrid_hid),
        ]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_fn(hid).to(DEVICE)
            print(f"\n  ── {name} (h={hid}, {model.param_count():,} params) ──")
            history = train_model(model, train_ds, EPOCHS, steps_per_epoch, label=f"{name} s={seed}")
            eval_result = evaluate_model(model, test_ds, label=f"{name} s={seed}")
            eval_result["history"] = history
            seed_results[name] = eval_result
            del model
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        all_results[f"seed_{seed}"] = seed_results

    # ── Aggregate results ──
    print(f"\n{'='*65}")
    print(f"  AGGREGATED RESULTS (Real Data)")
    print(f"{'='*65}")

    for model_name in ["homo", "hetero", "hybrid"]:
        accs = [all_results[f"seed_{s}"][model_name]["accuracy"]
                for s in [42, 123, 456][:NUM_SEEDS]]
        params = all_results[f"seed_{[42, 123, 456][0]}"][model_name]["params"]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"\n  {model_name:<10s}: {mean_acc:.4f} ± {std_acc:.4f}  ({params:,} params)")

        # Per-type
        for t in TYPE_NAMES:
            t_accs = [all_results[f"seed_{s}"][model_name]["per_type"].get(t, 0)
                       for s in [42, 123, 456][:NUM_SEEDS]]
            print(f"    {t:<12s}: {np.mean(t_accs):.4f} ± {np.std(t_accs):.4f}")

    # ── Print comparison ──
    homo_accs = [all_results[f"seed_{s}"]["homo"]["accuracy"] for s in [42, 123, 456][:NUM_SEEDS]]
    hetero_accs = [all_results[f"seed_{s}"]["hetero"]["accuracy"] for s in [42, 123, 456][:NUM_SEEDS]]
    hybrid_accs = [all_results[f"seed_{s}"]["hybrid"]["accuracy"] for s in [42, 123, 456][:NUM_SEEDS]]

    print(f"\n  Hetero vs Homo: {(np.mean(hetero_accs) - np.mean(homo_accs)) / np.mean(homo_accs) * 100:+.1f}%")
    print(f"  Hybrid vs Homo: {(np.mean(hybrid_accs) - np.mean(homo_accs)) / np.mean(homo_accs) * 100:+.1f}%")
    print(f"  Hetero wins: {sum(h > o for h, o in zip(hetero_accs, homo_accs))}/{NUM_SEEDS}")
    print(f"  Hybrid wins: {sum(h > o for h, o in zip(hybrid_accs, homo_accs))}/{NUM_SEEDS}")

    # ── Save ──
    out_dir = "hetmoe_realdata_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to {out_dir}/")

    # ── Generate figures ──
    print("\n  Generating figures...")
    generate_figures(all_results, out_dir)
    print(f"  Figures saved to {out_dir}/")


# ════════════════════════════════════════════════════════════
#  FIGURE GENERATION
# ════════════════════════════════════════════════════════════

def generate_figures(all_results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seeds = [k for k in all_results.keys()]
    models = ['homo', 'hetero', 'hybrid']
    colors = {'homo': '#4C72B0', 'hetero': '#DD8452', 'hybrid': '#55A868'}
    model_nice = {'homo': 'Homo (4×FFN)', 'hetero': 'Hetero', 'hybrid': 'Hybrid'}

    # ── Fig 1: Training loss curves (averaged across seeds) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for m in models:
        all_losses = []
        all_accs = []
        for s in seeds:
            h = all_results[s][m].get("history", {})
            if h:
                all_losses.append(h["loss"])
                all_accs.append(h["acc"])
        if all_losses:
            mean_loss = np.mean(all_losses, axis=0)
            std_loss = np.std(all_losses, axis=0)
            mean_acc = np.mean(all_accs, axis=0) * 100
            std_acc = np.std(all_accs, axis=0) * 100
            epochs = range(1, len(mean_loss) + 1)

            ax1.plot(epochs, mean_loss, label=model_nice[m], color=colors[m], linewidth=2)
            ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                           alpha=0.15, color=colors[m])
            ax2.plot(epochs, mean_acc, label=model_nice[m], color=colors[m], linewidth=2)
            ax2.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                           alpha=0.15, color=colors[m])

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss (Real Data)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Training Accuracy (%)', fontsize=11)
    ax2.set_title('Training Accuracy (Real Data)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_training_curves.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("    fig1_training_curves.png")

    # ── Fig 2: Per-type accuracy grouped bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5.5))
    types = TYPE_NAMES
    x = np.arange(len(types))
    width = 0.25

    for i, m in enumerate(models):
        type_means, type_stds = [], []
        for t in types:
            vals = [all_results[s][m]['per_type'][t] for s in seeds]
            type_means.append(np.mean(vals) * 100)
            type_stds.append(np.std(vals) * 100)
        ax.bar(x + (i - 1) * width, type_means, width, yerr=type_stds,
               capsize=4, label=model_nice[m],
               color=colors[m], edgecolor='black', linewidth=0.5, alpha=0.9)

    # Add improvement annotations
    for j, t in enumerate(types):
        homo_val = np.mean([all_results[s]['homo']['per_type'][t] for s in seeds]) * 100
        hetero_val = np.mean([all_results[s]['hetero']['per_type'][t] for s in seeds]) * 100
        if hetero_val > homo_val and homo_val > 0:
            ratio = hetero_val / homo_val
            ax.text(j, hetero_val + 4, f'{ratio:.1f}×', ha='center', fontsize=9,
                    color='#DD8452', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Per-Type Accuracy on Real Multimodal Data (~223K params)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_per_type_accuracy.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("    fig2_per_type_accuracy.png")

    # ── Fig 3: Overall accuracy bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    means, stds = [], []
    for m in models:
        accs = [all_results[s][m]['accuracy'] for s in seeds]
        means.append(np.mean(accs) * 100)
        stds.append(np.std(accs) * 100)

    bars = ax.bar(range(3), means, yerr=stds, capsize=8,
                  color=[colors[m] for m in models], edgecolor='black', linewidth=0.8,
                  width=0.6, alpha=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([model_nice[m] for m in models], fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy on Real Multimodal Data (~223K params)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Improvement annotations
    homo_mean = means[0]
    for i, m in enumerate(models[1:], 1):
        delta = (means[i] - homo_mean) / homo_mean * 100
        ax.annotate(f'+{delta:.1f}%', xy=(i, means[i] + stds[i] + 4),
                    fontsize=11, color=colors[m], fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_overall_accuracy.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("    fig3_overall_accuracy.png")

    # ── Fig 4: Per-seed robustness (dot plot) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        accs = [all_results[s][m]['accuracy'] * 100 for s in seeds]
        mean_acc = np.mean(accs)
        ax.scatter([i] * len(accs), accs, color=colors[m], s=80, zorder=5,
                   edgecolor='black', linewidth=0.8, alpha=0.8)
        ax.hlines(mean_acc, i - 0.2, i + 0.2, color=colors[m], linewidth=3, zorder=4)
        ax.text(i + 0.25, mean_acc, f'{mean_acc:.1f}%', fontsize=11, fontweight='bold',
                va='center', color=colors[m])

    ax.set_xticks(range(3))
    ax.set_xticklabels([model_nice[m] for m in models], fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Per-Seed Robustness on Real Data (~223K params)', fontsize=13, fontweight='bold')
    ax.set_ylim(35, 60)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_seed_robustness.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("    fig4_seed_robustness.png")

    # ── Fig 5: Radar chart ──
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    types_list = TYPE_NAMES
    angles = np.linspace(0, 2 * np.pi, len(types_list), endpoint=False).tolist()
    angles += angles[:1]

    for m in models:
        values = [np.mean([all_results[s][m]['per_type'][t] for s in seeds]) * 100 for t in types_list]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_nice[m], color=colors[m], markersize=6)
        ax.fill(angles, values, alpha=0.1, color=colors[m])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(types_list, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title('Per-Type Accuracy Profile\n(Real Multimodal Data, ~223K params)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_radar_profile.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("    fig5_radar_profile.png")


if __name__ == "__main__":
    main()
