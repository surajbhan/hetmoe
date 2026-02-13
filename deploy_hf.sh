#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Deploy HetMoE demo to HuggingFace Spaces
# ═══════════════════════════════════════════════════════════
#
#  Prerequisites:
#    1. pip install huggingface_hub
#    2. huggingface-cli login  (paste your HF token)
#    3. Trained model weights in hetmoe_realdata_results/
#       (run: python hetmoe_realdata.py)
#
#  Usage:
#    bash deploy_hf.sh                          # uses default: surajbhan/hetmoe
#    bash deploy_hf.sh your-username/hetmoe     # custom space name
#
# ═══════════════════════════════════════════════════════════

set -e

SPACE_NAME="${1:-surajbhan/hetmoe}"
DEPLOY_DIR=$(mktemp -d)

echo "═══════════════════════════════════════════"
echo "  Deploying HetMoE to HuggingFace Spaces"
echo "  Space: https://huggingface.co/spaces/$SPACE_NAME"
echo "  Temp dir: $DEPLOY_DIR"
echo "═══════════════════════════════════════════"

# ── Check prerequisites ──
if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

if [ ! -f "hetmoe_realdata_results/hetero_model.pt" ]; then
    echo "ERROR: Model weights not found. Run 'python hetmoe_realdata.py' first."
    exit 1
fi

if [ ! -f "hetmoe_realdata_results/norm_stats.json" ]; then
    echo "ERROR: norm_stats.json not found. Run 'python hetmoe_realdata.py' first."
    exit 1
fi

# ── Create HF Spaces README ──
cat > "$DEPLOY_DIR/README.md" << 'READMEEOF'
---
title: HetMoE - Heterogeneous Mixture-of-Experts
emoji: 🧠
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - mixture-of-experts
  - multimodal
  - heterogeneous-architectures
  - research
---

# HetMoE: Architecturally Heterogeneous Mixture-of-Experts

Interactive demo for the paper *"Architectural Heterogeneity in Mixture-of-Experts: Representational Complementarity Without Routing Specialization"*.

Upload an image, record audio, or provide a digit to see how the router distributes weight across four architecturally diverse experts (2D-CNN, Dilated-1D-CNN, FFT-Net, Self-Attention).

**GitHub**: [surajbhan/hetmoe](https://github.com/surajbhan/hetmoe)
**Author**: Surajbhan Satpathy, Yoctotta Technologies
READMEEOF

# ── Copy required files ──
echo "Copying files..."
cp app.py "$DEPLOY_DIR/"
cp requirements.txt "$DEPLOY_DIR/"

# Model weights and normalization stats
mkdir -p "$DEPLOY_DIR/hetmoe_realdata_results/examples"
cp hetmoe_realdata_results/hetero_model.pt "$DEPLOY_DIR/hetmoe_realdata_results/"
cp hetmoe_realdata_results/norm_stats.json "$DEPLOY_DIR/hetmoe_realdata_results/"

# Example files
if [ -d "hetmoe_realdata_results/examples" ]; then
    cp hetmoe_realdata_results/examples/* "$DEPLOY_DIR/hetmoe_realdata_results/examples/"
fi

echo ""
echo "Files staged in $DEPLOY_DIR:"
find "$DEPLOY_DIR" -type f | sed "s|$DEPLOY_DIR/||" | sort | sed 's/^/  /'
echo ""

# ── Upload to HuggingFace ──
echo "Uploading to HuggingFace Spaces..."
huggingface-cli upload "$SPACE_NAME" "$DEPLOY_DIR" . --repo-type space

echo ""
echo "═══════════════════════════════════════════"
echo "  Deployed successfully!"
echo "  https://huggingface.co/spaces/$SPACE_NAME"
echo "═══════════════════════════════════════════"

# Cleanup
rm -rf "$DEPLOY_DIR"
