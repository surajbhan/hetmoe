"""
HetMoE Interactive Demo — Hugging Face Spaces
================================================
Upload an image, record audio, or draw a digit to see how
architecturally heterogeneous MoE routes your input across
specialized experts (2D-CNN, Dilated-1D-CNN, FFT-Net, Self-Attention).
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

# ════════════════════════════════════════════════════════════
#  Model architecture (must match hetmoe_realdata.py exactly)
# ════════════════════════════════════════════════════════════

INPUT_DIM = 1024
NUM_CLASSES = 10


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
        h = x.unsqueeze(1)
        h = self.freq_conv(h).reshape(B, -1)
        return self.head(h)


class RelationalExpert(nn.Module):
    name_tag = "Relational"
    def __init__(self, din, hid, n_classes):
        super().__init__()
        self.n_tokens = int(np.sqrt(din))
        self.token_dim = self.n_tokens
        attn_dim = max((hid // 4 // 4) * 4, 16)
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
        w = self.router(x)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        output = (w.unsqueeze(-1) * expert_outs).sum(dim=1)
        avg_w = w.mean(dim=0)
        uniform = torch.ones_like(avg_w) / len(self.experts)
        bal_loss = self.bal_w * ((avg_w - uniform) ** 2).sum()
        ent_loss = -self.ent_w * (avg_w * torch.log(avg_w + 1e-8)).sum()
        return output, w, bal_loss + ent_loss


# ════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════

RESULTS_DIR = "hetmoe_realdata_results"
DEVICE = torch.device("cpu")  # Spaces runs on CPU

EXPERT_COLORS = {
    "Spatial": "#e74c3c",
    "Temporal": "#3498db",
    "Spectral": "#2ecc71",
    "Relational": "#9b59b6",
}

EXPERT_DESCRIPTIONS = {
    "Spatial": "2D Convolution (grid patterns)",
    "Temporal": "Dilated 1D Conv (sequences)",
    "Spectral": "FFT-based (frequency analysis)",
    "Relational": "Self-Attention (pairwise structure)",
}

# Label mappings
CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
SPEECH_LABELS = ["yes", "no", "up", "down", "left",
                 "right", "on", "off", "stop", "go"]
MNIST_LABELS = [str(i) for i in range(10)]


def load_model():
    """Load the trained heterogeneous MoE model."""
    model_path = os.path.join(RESULTS_DIR, "hetero_model.pt")
    if not os.path.exists(model_path):
        return None, "Model not found. Run `python hetmoe_realdata.py` first to train and save weights."

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    hid = ckpt["hidden_dim"]

    model = HetMoE(
        [SpatialExpert(INPUT_DIM, hid, NUM_CLASSES),
         TemporalExpert(INPUT_DIM, hid, NUM_CLASSES),
         SpectralExpert(INPUT_DIM, hid, NUM_CLASSES),
         RelationalExpert(INPUT_DIM, hid, NUM_CLASSES)],
        ["Spatial", "Temporal", "Spectral", "Relational"],
        INPUT_DIM
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, None


def load_norm_stats():
    """Load normalization statistics from training."""
    stats_path = os.path.join(RESULTS_DIR, "norm_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    # Fallback: zero mean, unit std
    return {k: {"mean": 0.0, "std": 1.0} for k in ["cifar", "speech", "spectral", "mnist"]}


# Load on startup
MODEL, LOAD_ERROR = load_model()
NORM_STATS = load_norm_stats()


# ════════════════════════════════════════════════════════════
#  Preprocessing functions
# ════════════════════════════════════════════════════════════

def preprocess_image(img):
    """Image → 1024-dim spatial input (grayscale 32x32)."""
    from PIL import Image
    if img is None:
        return None

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Convert to grayscale, resize to 32x32
    img = img.convert("L").resize((32, 32))
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.tensor(arr).view(1, -1)  # (1, 1024)

    # Normalize
    stats = NORM_STATS["cifar"]
    x = (x - stats["mean"]) / (stats["std"] + 1e-8)
    return x


def preprocess_audio(audio_tuple):
    """Audio → 1024-dim temporal input (resampled waveform)."""
    if audio_tuple is None:
        return None, None

    sr, waveform = audio_tuple

    # Convert to float tensor
    if isinstance(waveform, np.ndarray):
        wav = torch.tensor(waveform, dtype=torch.float32)
    else:
        wav = waveform.float()

    # If stereo, take first channel
    if wav.ndim > 1:
        wav = wav[:, 0] if wav.shape[1] < wav.shape[0] else wav[0]

    # Normalize waveform amplitude
    wav = wav / (wav.abs().max() + 1e-8)

    # Resample to 1024 via interpolation
    wav = F.interpolate(wav.unsqueeze(0).unsqueeze(0), size=1024,
                        mode='linear', align_corners=False).squeeze()

    # Temporal input
    x_temporal = wav.unsqueeze(0)  # (1, 1024)
    stats = NORM_STATS["speech"]
    x_temporal = (x_temporal - stats["mean"]) / (stats["std"] + 1e-8)

    # Spectral input (FFT)
    fft = torch.fft.rfft(wav, n=2046)
    mag = torch.abs(fft[:1024])
    x_spectral = mag.unsqueeze(0)  # (1, 1024)
    stats_s = NORM_STATS["spectral"]
    x_spectral = (x_spectral - stats_s["mean"]) / (stats_s["std"] + 1e-8)

    return x_temporal, x_spectral


def preprocess_digit(img):
    """Drawn digit → 1024-dim relational input (pairwise distance matrix)."""
    from PIL import Image
    if img is None:
        return None

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("L").resize((32, 32))
    arr = torch.tensor(np.array(img, dtype=np.float32) / 255.0)

    # Compute pairwise distance matrix (32 rows as nodes)
    nodes = arr  # (32, 32)
    diff = nodes.unsqueeze(0) - nodes.unsqueeze(1)  # (32, 32, 32)
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (32, 32)
    x = dist.view(1, -1)  # (1, 1024)

    stats = NORM_STATS["mnist"]
    x = (x - stats["mean"]) / (stats["std"] + 1e-8)
    return x


# ════════════════════════════════════════════════════════════
#  Inference + visualization
# ════════════════════════════════════════════════════════════

def make_routing_chart(weights, expert_names):
    """Create a matplotlib bar chart of expert routing weights."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [EXPERT_COLORS[n] for n in expert_names]
    bars = ax.barh(expert_names, weights, color=colors, edgecolor='black',
                   linewidth=0.8, height=0.6)

    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{w:.1%}', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, max(weights) * 1.3)
    ax.set_xlabel('Routing Weight', fontsize=11)
    ax.set_title('Expert Routing Weights', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def format_predictions(probs, label_names):
    """Format top-5 predictions as a dict for Gradio Label component."""
    top5_idx = probs.argsort()[-5:][::-1]
    return {label_names[i]: float(probs[i]) for i in top5_idx}


@torch.no_grad()
def predict(x, input_type, label_names):
    """Run inference and return predictions + routing chart."""
    if MODEL is None:
        return LOAD_ERROR, None, None

    logits, weights, _ = MODEL(x)
    probs = F.softmax(logits, dim=-1).squeeze().numpy()
    routing = weights.squeeze().numpy()

    predictions = format_predictions(probs, label_names)
    chart = make_routing_chart(routing, MODEL.names)

    # Expert analysis text
    top_expert_idx = routing.argmax()
    top_expert = MODEL.names[top_expert_idx]
    lines = [f"**Dominant Expert**: {top_expert} ({EXPERT_DESCRIPTIONS[top_expert]}) — {routing[top_expert_idx]:.1%} weight"]
    lines.append("")
    lines.append("| Expert | Architecture | Weight |")
    lines.append("|--------|-------------|--------|")
    for i, name in enumerate(MODEL.names):
        lines.append(f"| {name} | {EXPERT_DESCRIPTIONS[name]} | {routing[i]:.1%} |")
    lines.append("")
    lines.append(f"*Input type: {input_type}*")

    return predictions, chart, "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  Gradio interface functions
# ════════════════════════════════════════════════════════════

def classify_image(img):
    """Classify an uploaded image as a CIFAR-10 spatial input."""
    x = preprocess_image(img)
    if x is None:
        return None, None, "Please upload an image."
    return predict(x, "Spatial (CIFAR-10 style)", CIFAR10_LABELS)


def classify_audio(audio):
    """Classify audio — runs both temporal (waveform) and spectral (FFT) modes."""
    x_temporal, x_spectral = preprocess_audio(audio)
    if x_temporal is None:
        return None, None, None, None, "Please record or upload audio."

    pred_t, chart_t, analysis_t = predict(x_temporal, "Temporal (waveform)", SPEECH_LABELS)
    pred_s, chart_s, analysis_s = predict(x_spectral, "Spectral (FFT)", SPEECH_LABELS)
    return pred_t, chart_t, analysis_t, pred_s, chart_s, analysis_s


def classify_digit(img):
    """Classify a digit image as an MNIST relational input."""
    x = preprocess_digit(img)
    if x is None:
        return None, None, "Please upload or draw a digit."
    return predict(x, "Relational (MNIST style)", MNIST_LABELS)


# ════════════════════════════════════════════════════════════
#  Discover example files
# ════════════════════════════════════════════════════════════

def find_examples():
    """Find saved example files for each modality."""
    examples_dir = os.path.join(RESULTS_DIR, "examples")
    result = {"image": [], "audio": [], "digit": []}
    if not os.path.isdir(examples_dir):
        return result
    for f in sorted(os.listdir(examples_dir)):
        path = os.path.join(examples_dir, f)
        if f.startswith("cifar_") and f.endswith(".png"):
            result["image"].append(path)
        elif f.startswith("speech_") and f.endswith(".wav"):
            result["audio"].append(path)
        elif f.startswith("mnist_") and f.endswith(".png"):
            result["digit"].append(path)
    return result

EXAMPLES = find_examples()


# ════════════════════════════════════════════════════════════
#  Build Gradio App
# ════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(
        title="HetMoE: Heterogeneous Mixture-of-Experts Demo",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
# HetMoE: Architecturally Heterogeneous Mixture-of-Experts

This demo shows a trained **Heterogeneous MoE** model that uses four fundamentally different expert architectures:

| Expert | Architecture | Best For |
|--------|-------------|----------|
| **Spatial** | 2D Convolution | Images, grid-structured data |
| **Temporal** | Dilated 1D Conv | Audio waveforms, sequences |
| **Spectral** | FFT-based Network | Frequency spectra |
| **Relational** | Self-Attention | Pairwise structure, graphs |

Upload an image, record audio, or draw a digit to see how the router distributes weight across experts.

**Key insight**: The router learns to weight experts differently based on input structure, even though it receives no explicit modality label.
        """)

        if MODEL is None:
            gr.Markdown(f"**Model not loaded**: {LOAD_ERROR}")
        else:
            gr.Markdown(f"*Model loaded: {sum(p.numel() for p in MODEL.parameters()):,} parameters*")

        with gr.Tabs():
            # ── Tab 1: Image (Spatial) ──
            with gr.Tab("Image (Spatial)"):
                gr.Markdown("Upload any image. It will be converted to grayscale 32x32 and classified using CIFAR-10 labels.")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(type="pil", label="Upload Image")
                        img_btn = gr.Button("Classify", variant="primary")
                        if EXAMPLES["image"]:
                            gr.Examples(
                                examples=EXAMPLES["image"],
                                inputs=img_input,
                                label="Try these CIFAR-10 samples",
                            )
                    with gr.Column(scale=1):
                        img_pred = gr.Label(num_top_classes=5, label="Predictions (CIFAR-10)")
                        img_chart = gr.Image(label="Expert Routing")
                        img_analysis = gr.Markdown(label="Analysis")

                img_btn.click(classify_image, inputs=img_input,
                             outputs=[img_pred, img_chart, img_analysis])

            # ── Tab 2: Audio (Temporal + Spectral) ──
            with gr.Tab("Audio (Temporal + Spectral)"):
                gr.Markdown("Record or upload audio. The model processes it two ways: as a raw waveform (Temporal) and as an FFT spectrum (Spectral).")
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="Record or Upload Audio")
                        audio_btn = gr.Button("Classify", variant="primary")
                        if EXAMPLES["audio"]:
                            gr.Examples(
                                examples=EXAMPLES["audio"],
                                inputs=audio_input,
                                label="Try these Speech Commands samples",
                            )
                    with gr.Column(scale=1):
                        gr.Markdown("### Temporal (Waveform)")
                        audio_pred_t = gr.Label(num_top_classes=5, label="Temporal Predictions")
                        audio_chart_t = gr.Image(label="Temporal Routing")
                        audio_analysis_t = gr.Markdown()
                    with gr.Column(scale=1):
                        gr.Markdown("### Spectral (FFT)")
                        audio_pred_s = gr.Label(num_top_classes=5, label="Spectral Predictions")
                        audio_chart_s = gr.Image(label="Spectral Routing")
                        audio_analysis_s = gr.Markdown()

                audio_btn.click(classify_audio, inputs=audio_input,
                               outputs=[audio_pred_t, audio_chart_t, audio_analysis_t,
                                       audio_pred_s, audio_chart_s, audio_analysis_s])

            # ── Tab 3: Digit (Relational) ──
            with gr.Tab("Digit (Relational)"):
                gr.Markdown("Draw a digit or upload an image. It's converted to a pairwise distance matrix and classified using MNIST labels.")
                with gr.Row():
                    with gr.Column(scale=1):
                        digit_input = gr.Image(type="pil", label="Upload or Draw a Digit")
                        digit_btn = gr.Button("Classify", variant="primary")
                        if EXAMPLES["digit"]:
                            gr.Examples(
                                examples=EXAMPLES["digit"],
                                inputs=digit_input,
                                label="Try these MNIST samples",
                            )
                    with gr.Column(scale=1):
                        digit_pred = gr.Label(num_top_classes=5, label="Predictions (MNIST)")
                        digit_chart = gr.Image(label="Expert Routing")
                        digit_analysis = gr.Markdown(label="Analysis")

                digit_btn.click(classify_digit, inputs=digit_input,
                               outputs=[digit_pred, digit_chart, digit_analysis])

        gr.Markdown("""
---
**Paper**: [Architectural Heterogeneity in Mixture-of-Experts](https://github.com/surajbhan/hetmoe) |
**Author**: Surajbhan Satpathy, Yoctotta Technologies |
**Model**: ~223K parameters, trained on CIFAR-10 + Speech Commands + MNIST
        """)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
