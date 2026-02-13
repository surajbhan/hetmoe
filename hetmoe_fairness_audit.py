"""
APPLE-TO-APPLE FAIRNESS AUDIT
==============================
The v2 results have a core problem: param counts differ significantly.
  Homo:   689,800 params
  Hetero: 220,280 params  (3.1× fewer!)
  Hybrid: 352,280 params  (2.0× fewer)

We need to disentangle:
  (A) Is hetero better because of architectural diversity?
  (B) Is hetero better because smaller models regularize better on batch=64?
  (C) Is hetero better because it has the "right" total param budget?

This script runs CONTROLLED experiments:
  1. Param-matched at 3 budget levels (small/medium/large)
  2. Single-expert baselines (is MoE even helping?)
  3. Homo with same param count as hetero (shrink FFN hidden)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, time

# Import everything from v2
sys.path.insert(0, '.')

# ════════════════════════════════════════════════════════════
# Inline the needed classes (self-contained script)
# ════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name()}")
    else:
        d = torch.device("cpu")
        print("  CPU mode")
    return d


# ── Data Generator (same as v2) ──
class DataGenerator:
    def __init__(self, device):
        self.device = device
        self.diff_kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0

    @torch.no_grad()
    def gen_spatial(self, n):
        s = 32
        grids = torch.randn(n, 1, s, s, device=self.device) * 0.3
        for i in range(n):
            n_sources = torch.randint(2, 6, (1,)).item()
            for _ in range(n_sources):
                cx = torch.randint(2, s-2, (1,)).item()
                cy = torch.randint(2, s-2, (1,)).item()
                strength = (torch.rand(1, device=self.device).item() - 0.5) * 4.0
                yy, xx = torch.meshgrid(torch.arange(s, device=self.device),
                                         torch.arange(s, device=self.device), indexing='ij')
                blob = strength * torch.exp(-((xx-cx)**2 + (yy-cy)**2).float() / 8.0)
                grids[i, 0] += blob
        for _ in range(5):
            grids = F.conv2d(F.pad(grids, (1,1,1,1), mode='circular'), self.diff_kernel)
        final = F.conv2d(F.pad(grids, (1,1,1,1), mode='circular'), self.diff_kernel)
        c = s // 2
        targets = final[:, 0, c-4:c+4, c-4:c+4].mean(dim=(1,2))
        return grids.squeeze(1).reshape(n, -1), targets

    @torch.no_grad()
    def gen_temporal(self, n):
        T = 1024
        t = torch.arange(T+1, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n,-1)
        n_comp = torch.randint(3, 6, (n,))
        signal = torch.zeros(n, T+1, device=self.device)
        for i in range(n):
            for _ in range(n_comp[i].item()):
                freq = torch.rand(1, device=self.device).item() * 20 + 1
                amp = torch.rand(1, device=self.device).item() * 0.5 + 0.2
                phase = torch.rand(1, device=self.device).item() * 2 * np.pi
                signal[i] += amp * torch.sin(2*np.pi*freq*t[0]/T + phase)
        env_freq = torch.rand(n, 1, device=self.device) * 2 + 0.5
        envelope = 0.5 + 0.5 * torch.sin(2*np.pi*env_freq*t/T)
        signal = signal * envelope
        ar_coeffs = torch.rand(n, 5, device=self.device) * 0.15
        ar_coeffs = ar_coeffs / ar_coeffs.sum(dim=1, keepdim=True)
        for step in range(5, T+1):
            ar = (signal[:, step-5:step] * ar_coeffs).sum(dim=1)
            signal[:, step] = signal[:, step] + ar * 0.3
        signal = signal + torch.randn_like(signal) * 0.05
        weights = torch.exp(torch.linspace(-3, 0, 64, device=self.device))
        weights = weights / weights.sum()
        targets = (signal[:, -65:-1] * weights.unsqueeze(0)).sum(dim=1)
        return signal[:, :-1], targets

    @torch.no_grad()
    def gen_spectral(self, n):
        F_ = 1024
        spectra = torch.zeros(n, F_, device=self.device)
        dominant_freqs = torch.zeros(n, device=self.device)
        freqs_idx = torch.arange(1, F_+1, dtype=torch.float32, device=self.device)
        for i in range(n):
            spectra[i] = 0.1 / torch.sqrt(freqs_idx) * torch.randn(F_, device=self.device).abs()
            n_peaks = torch.randint(2, 5, (1,)).item()
            max_amp = 0
            for _ in range(n_peaks):
                center = torch.randint(10, F_-10, (1,)).item()
                width = torch.randint(2, 6, (1,)).item()
                amplitude = torch.rand(1, device=self.device).item() * 3.0 + 1.0
                bins = torch.arange(F_, dtype=torch.float32, device=self.device)
                peak = amplitude * torch.exp(-((bins - center)**2) / (2*width**2))
                spectra[i] += peak
                if amplitude > max_amp:
                    max_amp = amplitude
                    dominant_freqs[i] = center / F_
        spectra = spectra + torch.randn_like(spectra).abs() * 0.05
        return spectra, dominant_freqs

    @torch.no_grad()
    def gen_relational(self, n):
        s = 32
        points = torch.zeros(n, s, 2, device=self.device)
        targets = torch.zeros(n, device=self.device)
        for i in range(n):
            nc = torch.randint(2, 6, (1,)).item()
            centers = torch.rand(nc, 2, device=self.device) * 4 - 2
            spread = torch.rand(1, device=self.device).item() * 0.4 + 0.05
            assignments = torch.randint(0, nc, (s,))
            for j in range(s):
                points[i,j] = centers[assignments[j]] + torch.randn(2, device=self.device) * spread
            if nc > 1:
                cdiff = centers.unsqueeze(0) - centers.unsqueeze(1)
                cdist = torch.sqrt((cdiff**2).sum(-1) + 1e-8)
                mask = ~torch.eye(nc, dtype=torch.bool, device=self.device)
                targets[i] = cdist[mask].mean() / 4.0
        diff = points.unsqueeze(2) - points.unsqueeze(1)
        dist_matrix = torch.sqrt((diff**2).sum(-1) + 1e-8)
        return dist_matrix.reshape(n, -1), targets

    def mixed_batch(self, n):
        k = n // 4
        sizes = [k, k, k, n - 3*k]
        data, targets, labels = [], [], []
        for idx, (gen, sz) in enumerate([(self.gen_spatial, sizes[0]),
                                          (self.gen_temporal, sizes[1]),
                                          (self.gen_spectral, sizes[2]),
                                          (self.gen_relational, sizes[3])]):
            x, y = gen(sz)
            data.append(x); targets.append(y)
            labels.append(torch.full((sz,), idx, dtype=torch.long, device=self.device))
        data = torch.cat(data); targets = torch.cat(targets); labels = torch.cat(labels)
        perm = torch.randperm(len(data), device=self.device)
        return data[perm], targets[perm], labels[perm]


TYPE_NAMES = ["Spatial", "Temporal", "Spectral", "Relational"]

# ── Expert architectures (same as v2) ──
class FFNExpert(nn.Module):
    name_tag = "FFN"
    def __init__(self, din, hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(),
                                  nn.Linear(hid, hid//2), nn.GELU(),
                                  nn.Linear(hid//2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class SpatialExpert(nn.Module):
    name_tag = "Spatial2D"
    def __init__(self, din, hid):
        super().__init__()
        self.grid_side = int(np.sqrt(din))
        ch = max(hid//8, 8)
        self.conv = nn.Sequential(nn.Conv2d(1, ch, 5, padding=2), nn.GELU(),
                                   nn.Conv2d(ch, ch*2, 3, padding=1), nn.GELU(),
                                   nn.AdaptiveAvgPool2d(4))
        self.head = nn.Sequential(nn.Linear(ch*2*16, hid//2), nn.GELU(), nn.Linear(hid//2, 1))
    def forward(self, x):
        B = x.shape[0]; s = self.grid_side
        h = self.conv(x[:, :s*s].reshape(B, 1, s, s)).reshape(B, -1)
        return self.head(h).squeeze(-1)

class TemporalExpert(nn.Module):
    name_tag = "TempConv"
    def __init__(self, din, hid):
        super().__init__()
        ch = max(hid//8, 8)
        self.convs = nn.ModuleList([
            nn.Conv1d(1, ch, 7, padding=3, dilation=1),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4),
            nn.Conv1d(ch, ch, 3, padding=8, dilation=8)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(ch) for _ in range(4)])
        self.head = nn.Sequential(nn.Linear(ch, hid//2), nn.GELU(), nn.Linear(hid//2, 1))
    def forward(self, x):
        h = x.unsqueeze(1)
        for conv, norm in zip(self.convs, self.norms): h = F.gelu(norm(conv(h)))
        return self.head(h.mean(dim=-1)).squeeze(-1)

class SpectralExpert(nn.Module):
    name_tag = "Spectral"
    def __init__(self, din, hid):
        super().__init__()
        self.freq_conv = nn.Sequential(nn.Conv1d(1, hid//4, 7, padding=3), nn.GELU(),
                                        nn.Conv1d(hid//4, hid//4, 5, padding=2), nn.GELU(),
                                        nn.AdaptiveAvgPool1d(16))
        self.head = nn.Sequential(nn.Linear(hid//4*16, hid//2), nn.GELU(), nn.Linear(hid//2, 1))
    def forward(self, x):
        fft = torch.fft.rfft(x, dim=-1)
        mag = torch.abs(fft).unsqueeze(1)
        return self.head(self.freq_conv(mag).reshape(x.shape[0], -1)).squeeze(-1)

class RelationalExpert(nn.Module):
    name_tag = "Relational"
    def __init__(self, din, hid):
        super().__init__()
        self.n_tokens = int(np.sqrt(din)); self.token_dim = self.n_tokens
        attn_dim = max((hid//4 // 4) * 4, 16)
        self.proj = nn.Linear(self.token_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.head = nn.Sequential(nn.Linear(attn_dim, hid//2), nn.GELU(), nn.Linear(hid//2, 1))
    def forward(self, x):
        B = x.shape[0]
        tokens = x.reshape(B, self.n_tokens, self.token_dim)
        h = self.proj(tokens)
        attn_out, _ = self.attn(h, h, h)
        return self.head(self.norm(h + attn_out).mean(dim=1)).squeeze(-1)

# ── Router + MoE ──
class Router(nn.Module):
    def __init__(self, din, n_experts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden*2), nn.GELU(), nn.Linear(hidden*2, n_experts))
    def forward(self, x): return F.softmax(self.net(x), dim=-1)

class HetMoE(nn.Module):
    def __init__(self, experts, names, din, bal_w=0.05, ent_w=0.02):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.names = names
        self.router = Router(din, len(experts), 64)
        self.bal_w, self.ent_w = bal_w, ent_w
    def forward(self, x):
        w = self.router(x)
        outs = torch.stack([e(x) for e in self.experts], dim=1)
        output = (w * outs).sum(dim=1)
        avg = w.mean(dim=0); u = torch.ones_like(avg)/len(self.experts)
        aux = self.bal_w*((avg-u)**2).sum() - self.ent_w*(avg*torch.log(avg+1e-8)).sum()
        return output, w, aux
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training ──
def train_eval(model, datagen, epochs=400, bs=64, lr=1e-3, label=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    log_every = max(epochs // 4, 1)

    for ep in range(epochs):
        model.train()
        x, y, lab = datagen.mixed_batch(bs)
        pred, w, aux = model(x)
        loss = F.mse_loss(pred, y) + aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (ep+1) % log_every == 0:
            print(f"    [{label}] ep {ep+1}/{epochs} loss={F.mse_loss(pred,y).item():.4f}")

    # Eval
    model.eval()
    all_p, all_t, all_l, all_w = [], [], [], []
    with torch.no_grad():
        for _ in range(10):
            x, y, lab = datagen.mixed_batch(200)
            pred, w, _ = model(x)
            all_p.append(pred); all_t.append(y); all_l.append(lab); all_w.append(w)
    preds = torch.cat(all_p); tgts = torch.cat(all_t); labs = torch.cat(all_l); ws = torch.cat(all_w)
    overall = F.mse_loss(preds, tgts).item()
    per_type = {}
    routing = {}
    for t in range(4):
        m = labs == t
        if m.any():
            per_type[TYPE_NAMES[t]] = F.mse_loss(preds[m], tgts[m]).item()
            routing[TYPE_NAMES[t]] = ws[m].mean(dim=0).cpu().tolist()
    return {"overall": overall, "per_type": per_type, "params": model.param_count(),
            "routing": routing}


# ════════════════════════════════════════════════════════════
# FAIRNESS TESTS
# ════════════════════════════════════════════════════════════

def main():
    device = get_device()
    datagen = DataGenerator(device)
    DIN = 1024
    results = {}

    print("\n" + "="*65)
    print("  TEST 1: PARAM-MATCHED COMPARISON")
    print("  Find FFN hidden dim that gives ~220K params (same as hetero)")
    print("="*65)

    # Hetero at h=128 has 220,280 params. Find FFN hidden that matches.
    # FFN expert: din*hid + hid + hid*(hid//2) + hid//2 + (hid//2)*1 + 1
    # ≈ 1024*h + h + h*h/2 + h/2 + h/2 + 1
    # For 4 experts: 4 * (1024h + h²/2 + ...) + router
    # Let's just search
    for test_hid in [48, 52, 56, 60, 64, 68, 72]:
        m = HetMoE([FFNExpert(DIN, test_hid) for _ in range(4)],
                   ["F0","F1","F2","F3"], DIN).to(device)
        p = m.param_count()
        print(f"    FFN hidden={test_hid}: {p:,} params")
        del m

    # h=56 should be close to 220K. Let's use the closest.
    # Run param-matched homo
    for target_name, target_params, hid_guess in [
        ("Hetero-matched", 220280, 56),   # ~220K
    ]:
        # Verify
        test = HetMoE([FFNExpert(DIN, hid_guess) for _ in range(4)],
                      ["F0","F1","F2","F3"], DIN).to(device)
        actual_p = test.param_count()
        print(f"\n  Homo (h={hid_guess}): {actual_p:,} params (target: ~{target_params:,})")
        del test

    # Now run the actual comparison
    print("\n" + "="*65)
    print("  TEST 2: THREE-WAY PARAM-MATCHED (~220K params each)")
    print("="*65)

    # Small Homo (matched to Hetero's 220K)
    torch.manual_seed(42); np.random.seed(42)
    small_homo = HetMoE([FFNExpert(DIN, 56) for _ in range(4)],
                         ["FFN_0","FFN_1","FFN_2","FFN_3"], DIN).to(device)
    results["small_homo"] = train_eval(small_homo, datagen, label="Small Homo (h=56)")
    print(f"  Small Homo: {results['small_homo']['params']:,} params → MSE={results['small_homo']['overall']:.4f}")
    del small_homo; torch.cuda.empty_cache()

    # Hetero (h=128, naturally ~220K)
    torch.manual_seed(42); np.random.seed(42)
    hetero = HetMoE([SpatialExpert(DIN, 128), TemporalExpert(DIN, 128),
                      SpectralExpert(DIN, 128), RelationalExpert(DIN, 128)],
                     ["Spatial","Temporal","Spectral","Relational"], DIN).to(device)
    results["hetero"] = train_eval(hetero, datagen, label="Hetero (h=128)")
    print(f"  Hetero: {results['hetero']['params']:,} params → MSE={results['hetero']['overall']:.4f}")
    del hetero; torch.cuda.empty_cache()

    # Hybrid (h=96 to match ~220K... let's check)
    for test_hid in [72, 80, 88, 96]:
        m = HetMoE([FFNExpert(DIN, test_hid), SpatialExpert(DIN, test_hid),
                     TemporalExpert(DIN, test_hid), SpectralExpert(DIN, test_hid)],
                    ["FFN","Sp","Te","Spc"], DIN).to(device)
        print(f"    Hybrid hidden={test_hid}: {m.param_count():,} params")
        del m

    torch.manual_seed(42); np.random.seed(42)
    hybrid = HetMoE([FFNExpert(DIN, 80), SpatialExpert(DIN, 80),
                      TemporalExpert(DIN, 80), SpectralExpert(DIN, 80)],
                     ["FFN","Spatial","Temporal","Spectral"], DIN).to(device)
    results["hybrid_matched"] = train_eval(hybrid, datagen, label="Hybrid (h=80)")
    print(f"  Hybrid: {results['hybrid_matched']['params']:,} params → MSE={results['hybrid_matched']['overall']:.4f}")
    del hybrid; torch.cuda.empty_cache()

    print("\n" + "="*65)
    print("  TEST 3: PARAM-MATCHED AT ~400K")
    print("="*65)

    # Homo at ~400K
    torch.manual_seed(42); np.random.seed(42)
    homo_400 = HetMoE([FFNExpert(DIN, 80) for _ in range(4)],
                       ["FFN_0","FFN_1","FFN_2","FFN_3"], DIN).to(device)
    results["homo_400k"] = train_eval(homo_400, datagen, label="Homo ~400K")
    print(f"  Homo ~400K: {results['homo_400k']['params']:,} params → MSE={results['homo_400k']['overall']:.4f}")
    del homo_400; torch.cuda.empty_cache()

    # Hetero at ~400K (scale up hidden)
    torch.manual_seed(42); np.random.seed(42)
    hetero_400 = HetMoE([SpatialExpert(DIN, 226), TemporalExpert(DIN, 226),
                          SpectralExpert(DIN, 226), RelationalExpert(DIN, 226)],
                         ["Spatial","Temporal","Spectral","Relational"], DIN).to(device)
    results["hetero_400k"] = train_eval(hetero_400, datagen, label="Hetero ~400K")
    print(f"  Hetero ~400K: {results['hetero_400k']['params']:,} params → MSE={results['hetero_400k']['overall']:.4f}")
    del hetero_400; torch.cuda.empty_cache()

    # Hybrid at ~400K
    torch.manual_seed(42); np.random.seed(42)
    hybrid_400 = HetMoE([FFNExpert(DIN, 128), SpatialExpert(DIN, 128),
                          TemporalExpert(DIN, 128), SpectralExpert(DIN, 128)],
                         ["FFN","Spatial","Temporal","Spectral"], DIN).to(device)
    results["hybrid_400k"] = train_eval(hybrid_400, datagen, label="Hybrid ~400K")
    print(f"  Hybrid ~400K: {results['hybrid_400k']['params']:,} params → MSE={results['hybrid_400k']['overall']:.4f}")
    del hybrid_400; torch.cuda.empty_cache()

    print("\n" + "="*65)
    print("  TEST 4: PARAM-MATCHED AT ~700K")
    print("="*65)

    # Original Homo (h=128, 689K)
    torch.manual_seed(42); np.random.seed(42)
    homo_700 = HetMoE([FFNExpert(DIN, 128) for _ in range(4)],
                       ["FFN_0","FFN_1","FFN_2","FFN_3"], DIN).to(device)
    results["homo_700k"] = train_eval(homo_700, datagen, label="Homo ~700K")
    print(f"  Homo ~700K: {results['homo_700k']['params']:,} params → MSE={results['homo_700k']['overall']:.4f}")
    del homo_700; torch.cuda.empty_cache()

    # Hetero scaled to ~700K
    torch.manual_seed(42); np.random.seed(42)
    hetero_700 = HetMoE([SpatialExpert(DIN, 380), TemporalExpert(DIN, 380),
                          SpectralExpert(DIN, 380), RelationalExpert(DIN, 380)],
                         ["Spatial","Temporal","Spectral","Relational"], DIN).to(device)
    results["hetero_700k"] = train_eval(hetero_700, datagen, label="Hetero ~700K")
    print(f"  Hetero ~700K: {results['hetero_700k']['params']:,} params → MSE={results['hetero_700k']['overall']:.4f}")
    del hetero_700; torch.cuda.empty_cache()

    # Hybrid scaled to ~700K
    torch.manual_seed(42); np.random.seed(42)
    hybrid_700 = HetMoE([FFNExpert(DIN, 196), SpatialExpert(DIN, 196),
                          TemporalExpert(DIN, 196), SpectralExpert(DIN, 196)],
                         ["FFN","Spatial","Temporal","Spectral"], DIN).to(device)
    results["hybrid_700k"] = train_eval(hybrid_700, datagen, label="Hybrid ~700K")
    print(f"  Hybrid ~700K: {results['hybrid_700k']['params']:,} params → MSE={results['hybrid_700k']['overall']:.4f}")
    del hybrid_700; torch.cuda.empty_cache()

    print("\n" + "="*65)
    print("  TEST 5: SINGLE EXPERT BASELINES (no MoE overhead)")
    print("="*65)

    # Single FFN with same total params as hetero MoE
    torch.manual_seed(42); np.random.seed(42)
    class SingleModel(nn.Module):
        def __init__(self, expert):
            super().__init__()
            self.expert = expert
            self.names = ["single"]
        def forward(self, x):
            return self.expert(x), torch.ones(x.shape[0], 1, device=x.device), torch.tensor(0.0, device=x.device)
        def param_count(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Big single FFN
    single_ffn = SingleModel(FFNExpert(DIN, 256)).to(device)
    results["single_ffn"] = train_eval(single_ffn, datagen, label="Single FFN (h=256)")
    print(f"  Single FFN: {results['single_ffn']['params']:,} params → MSE={results['single_ffn']['overall']:.4f}")
    del single_ffn; torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("  APPLE-TO-APPLE RESULTS")
    print("="*70)

    print(f"\n  {'Budget':<12s} {'Model':<20s} {'Params':>10s} {'Overall':>9s}", end="")
    for t in TYPE_NAMES:
        print(f" {t[:7]:>8s}", end="")
    print()
    print("  " + "-"*80)

    groups = [
        ("~220K", ["small_homo", "hetero", "hybrid_matched"]),
        ("~400K", ["homo_400k", "hetero_400k", "hybrid_400k"]),
        ("~700K", ["homo_700k", "hetero_700k", "hybrid_700k"]),
        ("Single", ["single_ffn"]),
    ]

    for budget, keys in groups:
        for key in keys:
            r = results[key]
            name = key.replace("_", " ").title()
            print(f"  {budget:<12s} {name:<20s} {r['params']:>10,} {r['overall']:>9.4f}", end="")
            for t in TYPE_NAMES:
                print(f" {r['per_type'].get(t, -1):>8.4f}", end="")
            print()
        print("  " + "-"*80)

    # Save
    os.makedirs("hetmoe_audit", exist_ok=True)
    with open("hetmoe_audit/fairness_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to hetmoe_audit/fairness_results.json")


if __name__ == "__main__":
    main()
